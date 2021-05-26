import torch
import torch.nn as nn

from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_mse


class IncidenceEpidemicsGNN(GraphNeuralNetwork):
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
            1,
            1,
            lag=config.lag,
            nodeattr_size=1,
            edgeattr_size=edgeattr_size,
            out_act="identity",
            normalize=True,
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        return weighted_mse(y_true, y_pred, weights=weights)
