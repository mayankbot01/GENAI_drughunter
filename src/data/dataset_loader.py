import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]

def get_atom_features(atom):
    # Atomic number, degree, formal charge, hybridization, is_aromatic
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += [atom.GetIsAromatic()]
    return np.array(features, dtype=np.float32)

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        self.filename = filename
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
            
        node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(node_feats), dtype=torch.float)
        
        adj = Chem.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y, smiles=smiles)

def load_tox_data(csv_path):
    return MoleculeDataset(root='.', filename=csv_path)

if __name__ == "__main__":
    print(\"Dataset loader module ready with advanced featurization.\")
