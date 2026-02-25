import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

def one_hot_encoding(x, allowable_set):
    """One-hot encodes a value x based on an allowable set."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]

def get_atom_features(atom):
    """Generates a list of features for a given RDKit atom."""
    # Atomic symbol, degree, total number of Hs, implicit valence, is_aromatic
    features = []
    features += one_hot_encoding(atom.GetSymbol(), [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 
        'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 
        'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ])
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += [atom.GetIsAromatic()]
    return np.array(features, dtype=np.float32)

def visualize_molecule(smiles, output_path="molecule.png"):
    """Generates an image of a molecule from its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Draw.MolToFile(mol, output_path)
        print(f"Molecule saved to {output_path}")
    else:
        print("Invalid SMILES.")

def plot_training_history(history, title="Training Loss"):
    """Plots training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()Add featurization functions to helpers.py

if __name__ == "__main__":
    print("Helpers module ready with featurization logic.")
