import torch

import torch.nn as nn

from collections import OrderedDict

from torch_geometric.nn import SAGEConv


class ConvAct(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, activation=nn.ReLU()):
        super().__init__()
        self.conv = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.activation = activation
        
    def forward(self, graph):
        graph['x'] = self.conv(graph['x'], graph['edge_index'])
        graph['x'] = self.activation(graph['x'])
        
        return graph

class CustomSAGE(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_conv_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.conv_layers = OrderedDict()
        self.activation = nn.ReLU()
        for i in range(0, self.num_conv_layers*2, 2):
            self.conv_layers[str(i)] = ConvAct(in_channels=self.hidden_dim, out_channels=self.hidden_dim, 
                                               activation = self.activation)
        self.conv_layers = nn.Sequential(self.conv_layers)
        self.last = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def forward(self, graph):
        graph['x'] = self.emb_layer(graph['x'])
        graph = self.conv_layers(graph)
        graph['x'] = self.last(graph['x'])
        probs = nn.functional.softmax(graph['x'], dim=1)
        
        return probs