import torch
import torch.nn as nn

from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.loss import weighted_mse


class ReactionDiffusionGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        if "is_weighted" in config.__dict__ and config.is_weighted:
            edgeattr_size = 1
        else:
            edgeattr_size = 0
        self.num_states = config.num_states
        self.alpha = config.alpha
        GraphNeuralNetwork.__init__(
            self,
            config.num_states,
            config.num_states,
            edgeattr_size=edgeattr_size,
            lag=config.lag,
            normalize=True,
            config=config,
            **kwargs
        )

    def loss(self, y_true, y_pred, weights):
        l1 = weighted_mse(y_true, y_pred, weights=weights)
        sizes_true = torch.sum(y_true, axis=-1)
        sizes_pred = torch.sum(y_pred, axis=-1)
        l2 = torch.sum(weights * torch.abs(sizes_true - sizes_pred))
        return self.alpha[0] * l1 + self.alpha[0] * l2
