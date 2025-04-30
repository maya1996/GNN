import os
import torch
from torch_geometric.data import Dataset, Data
from ase.io import read
from ase.data import atomic_numbers
import numpy as np

class AtomicChargeDataset(Dataset):
    def __init__(self, root, poscar_dir="POSCAR", charge_dir="CHARGE", transform=None, threshold=3.0):
        super().__init__(root, transform)
        self.poscar_dir = os.path.join(root, poscar_dir)
        self.charge_dir = os.path.join(root, charge_dir)
        self.threshold = threshold
        self.file_ids = self._get_file_ids()

    def _get_file_ids(self):
        files = sorted([f for f in os.listdir(self.poscar_dir) if f.startswith("CONFIG_") and f.endswith(".POSCAR")])
        return [f.split("_")[1].split(".")[0] for f in files]

    def len(self):
        return len(self.file_ids)

    def get(self, idx):
        file_id = self.file_ids[idx]
        poscar_path = os.path.join(self.poscar_dir, f"CONFIG_{file_id}.POSCAR")
        charge_path = os.path.join(self.charge_dir, f"CHARGE_{file_id}")

        atoms = read(poscar_path, format='vasp')
        positions = atoms.get_positions()
        atom_types = torch.tensor([atomic_numbers[sym] for sym in atoms.get_chemical_symbols()], dtype=torch.float)

        with open(charge_path, "r") as f:
            charges = torch.tensor([float(line.strip()) for line in f], dtype=torch.float)

        dist_matrix = atoms.get_all_distances(mic=True)
        edge_index = []
        edge_attr = []

        for i in range(len(atom_types)):
            for j in range(len(atom_types)):
                if i != j and dist_matrix[i, j] <= self.threshold:
                    edge_index.append([i, j])
                    edge_attr.append([dist_matrix[i, j]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(
            x=atom_types.view(-1, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=charges.view(-1, 1),
            pos=torch.tensor(positions, dtype=torch.float)
        )


if __name__ == "__main__":
    dataset = AtomicChargeDataset(root=".")

    data_list = []
    failed = []

    for i in range(len(dataset)):
        try:
            data = dataset[i]
            data_list.append(data)
        except Exception as e:
            failed.append((i, str(e)))

    torch.save(data_list, "atomic_charge_dataset_full.pt")
    print(f"Saved {len(data_list)} molecules to 'atomic_charge_dataset_full.pt'")
    print(f"Skipped {len(failed)} files due to errors.")
    for idx, error in failed:
        print(f"  - Index {idx}: {error}")