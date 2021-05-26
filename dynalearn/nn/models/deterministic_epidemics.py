import torch
import torch.nn as nn

from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_cross_entropy


class DeterministicEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        if "is_weighted" in config.__dict__ and config.is_weighted:
            edgeattr_size = 1
        else:
            edgeattr_size = 0
        self.num_states = config.num_states
        GraphNeuralNetwork.__init__(
            self,
            config.num_states,
            config.num_states,
            lag=config.lag,
            nodeattr_size=1,
            edgeattr_size=edgeattr_size,
            out_act="softmax",
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_cross_entropy(y_true, y_pred, weights=weights)
