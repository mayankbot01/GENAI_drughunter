import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ToxGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(ToxGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch) # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def get_model(num_features=78, hidden_dim=64):
    """Returns the ToxGNN model. Default num_features=78 for advanced featurization."""
    return ToxGNN(num_node_features=num_features, hidden_channels=hidden_dim)

if __name__ == "__main__":
    model = get_model()
    print("GNN Model initialized.")
    print(model)
