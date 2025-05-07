import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

def data_preprocessing():
    data_list = torch.load('atomic_charge_dataset_full.pt')
    print(f"📦 Loaded {len(data_list)} graphs")

# Split dataset
    train_data = data_list[:int(0.8 * len(data_list))]
    val_data = data_list[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
    test_data = data_list[int(0.9 * len(data_list)):]
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    return train_loader, val_loader, test_loader

class Hyperparameters:
    def __init__(self):
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
    

# Define GCN model
class GCNRegressor(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x.view(-1)


# Training function
def train(train_loader, model, optimizer, val_loader, hp, device):
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_losses, val_losses, acc_val = [], [], []
    for epoch in range(1, hp.num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        acc_val.append(val_acc)
        print(f"Epoch{epoch+1}: Train loss:{avg_train_loss:.4f} Validation Loss:{val_loss:.4f} Accuracy:{val_acc:.2f}%")
        plot_losses(train_losses, val_losses)



# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    true, pred = [], []
    criterion = nn.MSELoss()
   
  
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            total_loss += loss.item()
            true.append(batch.y.view(-1).detach().cpu())
            pred.append(out.detach().cpu())

    true = torch.cat(true).numpy()
    pred = torch.cat(pred).numpy()
    avg_test_loss = total_loss / len(loader)
    acc = r2_score(true, pred)
    return avg_test_loss, acc
           

# Plotting function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    hp = Hyperparameters()
    train_loader, val_loader, test_loader = data_preprocessing()
    model = GCNRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    train(train_loader, model, optimizer, val_loader, hp, device)
    print("✅ Model training complete.")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"📉 Test Loss: {test_loss:.4f},R² Score: {test_acc:.4f}")