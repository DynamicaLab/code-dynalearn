import numpy as np
import time
import torch

from dynalearn.dynamics.stochastic_epidemics import StochasticEpidemics
from dynalearn.nn.models import StochasticEpidemicsGNN
from dynalearn.config import Config


class GNNSEDynamics(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        self.config = config or Config(**kwargs)
        StochasticEpidemics.__init__(self, config, config.num_states)
        self.nn = StochasticEpidemicsGNN(self.config)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert x.ndim == 2
        assert x.shape[-1] == self.lag
        x = self.nn.transformers["t_inputs"].forward(x)
        g = self.nn.transformers["t_networks"].forward(self.network)
        y = self.nn.transformers["t_targets"].backward(self.nn.forward(x, g))
        return y.cpu().detach().numpy()

    def number_of_infected(self, x):
        return np.inf

    def nearly_dead_state(self, **kwargs):
        return self.initial_state()
