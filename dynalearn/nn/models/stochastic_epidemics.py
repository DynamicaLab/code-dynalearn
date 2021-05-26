import torch
import torch.nn as nn

from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_cross_entropy


class StochasticEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        self.num_states = config.num_states
        if "is_weighted" in config.__dict__ and config.is_weighted:
            edgeattr_size = 1
        else:
            edgeattr_size = 0
        GraphNeuralNetwork.__init__(
            self,
            1,
            config.num_states,
            edgeattr_size=edgeattr_size,
            lag=config.lag,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)
