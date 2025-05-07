import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

def data_preprocessing():
    data_list = torch.load('atomic_charge_dataset_full.pt')
    print(f"📦 Loaded {len(data_list)} graphs")
    torch.manual_seed(42)
    np.random.seed(42)
    np.random.shuffle(data_list)

# Split dataset
    train_size= int(0.8 * len(data_list))
    val_size = int(0.1 * len(data_list))
    test_size = len(dataset) - train_size - val_size
  
    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]


    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    return train_loader, val_loader, test_loader

def get_in_channels(dataset):
    for data in dataset:
        return data.num_node_features

class Hyperparameters:
    def __init__(self):
        self.num_epochs = 250
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
    

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1, dropout=0.3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pool to graph-level representation
        x = self.lin(x)
        return x

# Training function
def train(train_loader, model, optimizer, val_loader, hp, device):
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    train_losses, val_losses, acc_val = [], [], []
    for epoch in range(1, hp.num_epochs + 1):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(out.view(-1), data.y.view(-1))
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
    
    plot_chart(range(1, hp.num_epochs + 1), train_losses, val_losses, acc_val)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []
   
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            true.append(data.y.view(-1).cpu())
            pred.append(out.view(-1).cpu())

    true = torch.cat(true)
    pred = torch.cat(pred)

    loss = F.l1_loss(true, pred).item()
    total_loss += loss.item()
    acc = r2_score(true, pred)

    return loss.item(), acc
           

# Plotting function
def plot_chart(epochs, train_losses, val_losses, accuracies):
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

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.savefig("loss_acc.png")
    plt.legend()
    plt.show()
    plt.close()



if __name__ == "__main__":
    hp = Hyperparameters()
    train_loader, val_loader, test_loader = data_preprocessing()
    node_features = get_in_channels(train_loader.dataset)
    model = GCN(node_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate,weight_decay=hp.weight_decay)
    train(train_loader, model, optimizer, val_loader, hp, device)
    print("✅ Model training complete.")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"📉 Test Loss: {test_loss:.4f}, Test_Accuarcy: {test_acc:.2f}%")