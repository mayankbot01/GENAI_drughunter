import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.data = pd.read_csv(filename)
        
    def len(self):
        return len(self.data)

    def get(self, idx):
        smiles = self.data.iloc[idx]['smiles']
        label = self.data.iloc[idx]['toxicity']
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Data()
            
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append([atom.GetAtomicNum()])
        x = torch.tensor(node_feats, dtype=torch.float)
        
        adj = Chem.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y, smiles=smiles)

def load_tox_data(csv_path):
    return MoleculeDataset(root='.', filename=csv_path)

if __name__ == "__main__":
    print(\"Dataset loader module ready.\")
