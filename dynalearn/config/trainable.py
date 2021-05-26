from .config import Config
from .util import OptimizerConfig
from dynalearn.nn.optimizers import *


class TrainableConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.name = "GNNSEDynamics"
        cls.num_states = 2
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [2]
        cls.gnn_channels = 2
        cls.out_channels = [2]
        cls.heads = 1
        cls.concat = False
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sis(cls):
        cls = cls()
        cls.name = "GNNSEDynamics"
        cls.gnn_name = "DynamicsGATConv"
        cls.type = "linear"

        cls.num_states = 2
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.name = "GNNSEDynamics"
        cls.gnn_name = "DynamicsGATConv"
        cls.type = "linear"

        cls.num_states = 2
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.name = "GNNSEDynamics"
        cls.gnn_name = "DynamicsGATConv"
        cls.type = "linear"

        cls.num_states = 4
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.heads = 4
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.name = "GNNDEDynamics"
        cls.gnn_name = "DynamicsGATConv"
        cls.type = "linear"

        cls.num_states = 3
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.weighted = True

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.node_activation = "relu"
        cls.edge_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32, 32]
        cls.gnn_channels = 32
        cls.node_channels = [4, 4]
        cls.edge_channels = [4, 4]
        cls.edge_gnn_channels = 4
        cls.out_channels = [32, 32, 32]
        cls.heads = 8
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def incsir(cls):
        cls = cls()
        cls.name = "GNNIncidenceDynamics"
        cls.gnn_name = "DynamicsGATConv"
        cls.type = "linear"

        cls.num_states = 1
        cls.lag = 1
        cls.lagstep = 1

        cls.optimizer = OptimizerConfig.default()

        cls.weighted = True

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.node_activation = "relu"
        cls.edge_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [4, 8, 16]
        cls.gnn_channels = 32
        cls.node_channels = [4, 4]
        cls.edge_channels = [4, 8]
        cls.edge_gnn_channels = 8
        cls.out_channels = [16, 8, 4]
        cls.heads = 1
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def kapoor(cls):
        cls = cls()
        cls.name = "KapoorDynamics"
        cls.lag = 7
        cls.lagstep = 1
        cls.num_states = 1
        cls.optimizer = OptimizerConfig.default()
        return cls
