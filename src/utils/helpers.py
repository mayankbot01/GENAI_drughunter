import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

def visualize_molecule(smiles, output_path=\"molecule.png\"):
    \"\"\"Generates an image of a molecule from its SMILES string.\"\"\"
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Draw.MolToFile(mol, output_path)
        print(f\"Molecule saved to {output_path}\")
    else:
        print(\"Invalid SMILES.\")

def plot_training_history(history, title=\"Training Loss\"):
    \"\"\"Plots training history.\"\"\"
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == \"__main__\":
    print(\"Helpers module ready.\")
