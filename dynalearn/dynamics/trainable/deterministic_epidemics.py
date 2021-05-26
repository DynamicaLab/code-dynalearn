import networkx as nx
import numpy as np
import time
import torch

from dynalearn.dynamics.deterministic_epidemics import (
    DeterministicEpidemics,
    WeightedDeterministicEpidemics,
    MultiplexDeterministicEpidemics,
    WeightedMultiplexDeterministicEpidemics,
)
from dynalearn.nn.models import DeterministicEpidemicsGNN
from dynalearn.nn.optimizers import get as get_optimizer
from dynalearn.util import to_edge_index
from dynalearn.config import Config


class SGNNDEDynamics(DeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        DeterministicEpidemics.__init__(self, config, config.num_states)
        self.nn = DeterministicEpidemicsGNN(config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def is_dead(self):
        return False

    def update(self, x):
        raise ValueError("This method is invalid for Trainable models")

    def infection_rate(self, x):
        raise ValueError("This method is invalid for Trainable models")

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


class WGNNDEDynamics(SGNNDEDynamics, WeightedDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedDeterministicEpidemics.__init__(self, config, config.num_states)
        SGNNDEDynamics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsGNN(config)


class MGNNDEDynamics(SGNNDEDynamics, MultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        MultiplexDeterministicEpidemics.__init__(self, config, config.num_states)
        SGNNDEDynamics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsGNN(config)


class WMGNNDEDynamics(SGNNDEDynamics, WeightedMultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        WeightedMultiplexDeterministicEpidemics.__init__(
            self, config, config.num_states
        )
        SGNNDEDynamics.__init__(self, config=config, **kwargs)
        self.nn = DeterministicEpidemicsGNN(config)


def GNNDEDynamics(config=None, **kwargs):
    config = config or Config(**kwargs)
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WMGNNDEDynamics(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WGNNDEDynamics(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MGNNDEDynamics(config=config, **kwargs)
    else:
        return SGNNDEDynamics(config=config, **kwargs)
