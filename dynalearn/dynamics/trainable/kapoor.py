import networkx as nx
import numpy as np
import time
import torch

from dynalearn.dynamics.incidence import (
    IncidenceEpidemics,
    WeightedIncidenceEpidemics,
    MultiplexIncidenceEpidemics,
    WeightedMultiplexIncidenceEpidemics,
)
from dynalearn.nn.models import Kapoor2020GNN
from dynalearn.nn.optimizers import get as get_optimizer
from dynalearn.util import to_edge_index
from dynalearn.config import Config


class KapoorDynamics(IncidenceEpidemics):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        IncidenceEpidemics.__init__(self, config)
        self.nn = Kapoor2020GNN(config)
        self.lag = self.nn.lag
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def is_dead(self):
        return False

    def update(self, x):
        raise ValueError("This method is invalid for Trainable models.")

    def infection_rate(self, x):
        raise ValueError("This method is invalid for Trainable models.")

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert x.ndim == 3
        assert x.shape[1] == self.num_states
        assert x.shape[2] == self.lag
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.cpu().detach().numpy()
