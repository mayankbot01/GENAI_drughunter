import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

try:
    from src.models.gnn_model import get_model
    from src.data.dataset_loader import load_tox_data
except ImportError:
    from ..models.gnn_model import get_model
    from ..data.dataset_loader import load_tox_data

def train():
    """Simple training loop for the ToxGNN model."""
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.001
    
    # Load dataset (assuming a CSV named 'tox_data.csv' exists in root or data folder)
    try:
        dataset = load_tox_data('data/tox_data.csv')
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}. Using dummy path for demonstration.")
        # In a real scenario, the user would provide the correct path
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model with 78 features
    model = get_model(num_features=78)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in tqdm(loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
    
    # Save the model
    torch.save(model.state_path(), 'models/tox_gnn_model.pth')
    print("Training complete. Model saved to models/tox_gnn_model.pth")

if __name__ == "__main__":
    train()
