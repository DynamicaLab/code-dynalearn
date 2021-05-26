import numpy as np

from .base import StochasticEpidemics
from dynalearn.dynamics.activation import independent
from dynalearn.config import Config


class SIS(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        StochasticEpidemics.__init__(self, config, 2)
        self.infection = config.infection
        self.recovery = config.recovery

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        ltp = np.zeros((x.shape[0], self.num_states))
        p = independent(self.neighbors_state(x)[1], self.infection)
        q = self.recovery
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 1, 0] = q
        ltp[x == 1, 1] = 1 - q
        return ltp

    def number_of_infected(self, x):
        return np.sum(x == 1)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 1
        return x


class SIR(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        StochasticEpidemics.__init__(self, config, 3)
        self.infection = config.infection
        self.recovery = config.recovery

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        ltp = np.zeros((x.shape[0], self.num_states))
        p = independent(self.neighbors_state(x)[1], self.infection)
        q = self.recovery
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 0, 2] = 0
        ltp[x == 1, 0] = 0
        ltp[x == 1, 1] = 1 - q
        ltp[x == 1, 2] = q
        ltp[x == 2, 0] = 0
        ltp[x == 2, 1] = 0
        ltp[x == 2, 2] = 1
        return ltp

    def number_of_infected(self, x):
        return np.sum(x == 1)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 1
        return x
