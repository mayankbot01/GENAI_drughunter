import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    """Fuses GNN embeddings with molecular descriptors."""
    def __init__(self, gnn_out_dim, descriptor_dim, hidden_dim=32):
        super(MultiModalFusion, self).__init__()
        self.fc1 = nn.Linear(gnn_out_dim + descriptor_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, gnn_x, descriptors):
        # Concatenate GNN output and descriptors
        x = torch.cat([gnn_x, descriptors], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_fusion_model(gnn_dim=64, desc_dim=10):
    """Returns the MultiModalFusion model."""
    return MultiModalFusion(gnn_out_dim=gnn_dim, descriptor_dim=desc_dim)

if __name__ == "__main__":
    model = get_fusion_model()
    print("Fusion Model initialized.")
    print(model)
