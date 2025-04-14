import torch
from torch import nn
from torch_geometric.nn import LightGCN

def get_lightgcn_model(num_nodes, embedding_dim=128, num_layers=2):
    return LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
