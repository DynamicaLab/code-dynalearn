import torch
import torch.nn as nn

from dynalearn.config import Config
from .model import Model
from torch_geometric.nn import GCNConv
from .dgat import DynamicsGATConv
from torch.nn.init import kaiming_normal_
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.transformers import BatchNormalizer
from dynalearn.nn.loss import weighted_mse


class Kapoor2020GNN(Model):
    def __init__(self, config=None, **kwargs):
        Model.__init__(self, config=config, **kwargs)
        self.lag = 7
        self.lagstep = 1
        self.nodeattr_size = 1
        self.num_states = 1
        self.in_layers = nn.Linear(self.num_states * self.lag + self.nodeattr_size, 64)
        self.gnn1 = GCNConv(64, 32)
        self.gnn2 = GCNConv(32, 32)
        self.out_layers = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

        self.transformers = BatchNormalizer(input_size=1, target_size=1, node_size=1)

        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, network_attr):
        edge_index, edge_attr, node_attr = network_attr
        x = x.view(-1, self.num_states * self.lag)
        x = torch.cat([x, node_attr], axis=-1)
        x = self.dropout(self.activation(self.in_layers(x)))
        x = self.dropout(self.activation(self.gnn1(x, edge_index)))
        x = self.dropout(self.activation(self.gnn2(x, edge_index)))
        x = self.out_layers(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        initialize_inplace(self.in_layers.weight)
        if self.in_layers.bias is not None:
            self.in_layers.bias.data.fill_(0)

        initialize_inplace(self.out_layers.weight)
        if self.out_layers.bias is not None:
            self.out_layers.bias.data.fill_(0)
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()

    def loss(self, y_true, y_pred, weights):
        return weighted_mse(y_true, y_pred)
