import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



import pickle
import random
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Graph(torch.nn.Module):
    def __init__(self, node_feature_dim, output_dim):
        super(Graph, self).__init__()
        self.conv1 = GATConv(node_feature_dim, 64, heads=3, concat=False, dropout=0.5)
        self.conv2 = GATConv(64, 64, heads=3, concat=False)
        self.linear = torch.nn.Linear(64, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        logit = self.linear(x)
        return logit, x