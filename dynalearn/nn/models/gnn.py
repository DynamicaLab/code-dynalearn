import numpy as np
import torch
import torch.nn as nn

from torch.nn import Parameter, Sequential, Linear, Identity
from .dgat import DynamicsGATConv
from .model import Model
from .util import (
    get_in_layers,
    get_out_layers,
    reset_layer,
    ParallelLayer,
    LinearLayers,
)
from torch.nn.init import kaiming_normal_
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from dynalearn.nn.transformers import BatchNormalizer
from dynalearn.config import Config


class GraphNeuralNetwork(Model):
    def __init__(
        self,
        in_size,
        out_size,
        lag=1,
        nodeattr_size=0,
        edgeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        Model.__init__(self, config=config, **kwargs)

        self.in_size = self.config.in_size = in_size
        self.out_size = self.config.out_size = out_size
        self.out_act = self.config.out_act = out_act
        self.lag = self.config.lag = lag
        self.nodeattr_size = nodeattr_size
        self.edgeattr_size = edgeattr_size

        if self.nodeattr_size > 0 and "node_channels" in self.config.__dict__:
            self.node_layers = LinearLayers(
                [self.nodeattr_size] + self.config.node_channels,
                self.config.node_activation,
                self.config.bias,
            )
        else:
            self.node_layers = Sequential(Identity())
            self.config.node_channels = [self.nodeattr_size]

        if self.edgeattr_size > 0 and "edge_channels" in self.config.__dict__:
            template = lambda: LinearLayers(
                [self.edgeattr_size] + self.config.edge_channels,
                self.config.edge_activation,
                self.config.bias,
            )
            if self.config.is_multiplex:
                self.edge_layers = ParallelLayer(
                    template, keys=self.config.network_layers
                )
            else:
                self.edge_layers = template()
        else:
            self.edge_layers = Sequential(Identity())

        self.in_layers = get_in_layers(self.config)
        self.gnn_layer = get_gnn_layer(self.config)
        self.out_layers = get_out_layers(self.config)

        if normalize:
            input_size = in_size
            target_size = out_size
        else:
            input_size = 0
            target_size = 0
        self.transformers = BatchNormalizer(
            input_size=input_size,
            target_size=target_size,
            edge_size=edgeattr_size,
            node_size=nodeattr_size,
            layers=self.config.network_layers,
        )

        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, network_attr):
        edge_index, edge_attr, node_attr = network_attr
        x = self.in_layers(x)
        node_attr = self.node_layers(node_attr)
        edge_attr = self.edge_layers(edge_attr)

        x = self.merge_nodeattr(x, node_attr)
        if self.config.gnn_name == "DynamicsGATConv":
            x = self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.gnn_layer(x, edge_index)
        if isinstance(x, tuple):
            x = x[0]
        x = self.out_layers(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        reset_layer(self.edge_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()

    def merge_nodeattr(self, x, node_attr):
        if node_attr is None:
            return x
        # print(x.shape, node_attr.shape)
        assert x.shape[0] == node_attr.shape[0]
        n = x.shape[0]
        return torch.cat([x, node_attr.view(n, -1)], dim=-1)
