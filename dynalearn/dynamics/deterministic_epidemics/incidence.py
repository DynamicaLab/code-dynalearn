import numpy as np
from .simple import (
    SimpleDSIR,
    WeightedDSIR,
    MultiplexDSIR,
    WeightedMultiplexDSIR,
)
from dynalearn.config import Config

EPSILON = 0.0


class SimpleIncSIR(SimpleDSIR):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.latent_state = None
        self._num_states = 1

    def initial_state(self, init_param=None, density=None, squeeze=True):
        if init_param is None:
            init_param = self.init_param
        if not isinstance(init_param, (np.ndarray, list)):
            init_param = np.random.rand()

        x = np.zeros([self.num_nodes, self.num_states])
        self.population = self.init_population(density=density)
        for i, n in enumerate(self.population):
            x[i] = np.random.binomial(n, init_param) / n

        self.latent_state = np.zeros((x.shape[0], 3))
        self.latent_state[:, 0] = 1 - x.squeeze()
        self.latent_state[:, 1] = x.squeeze()
        x = x.reshape(*x.shape, 1).repeat(self.lag, -1)
        if squeeze:
            return x.squeeze()
        else:
            return x

    def predict(self, x):
        if x.ndim == 3:
            x = x[:, :, -1]
        x = x.squeeze()
        if self.latent_state is None:
            self.latent_state = np.zeros((x.shape[0], 3))
            self.latent_state[:, 0] = 1 - x / self.population
            self.latent_state[:, 1] = x / self.population
        self.latent_state[:, 0] -= x / self.population
        self.latent_state[:, 1] += x / self.population
        p = super().predict(self.latent_state)
        self.latent_state = p * 1
        current_i = self.latent_state[:, 1]
        future_i = p[:, 1] * self.population
        y = (future_i - current_i).reshape(-1, 1)
        return y


class WeightedIncSIR(SimpleIncSIR, WeightedDSIR):
    def __init__(self, config=None, **kwargs):
        WeightedDSIR.__init__(self, config=config, **kwargs)
        SimpleIncSIR.__init__(self, config=config, **kwargs)


class MultiplexIncSIR(SimpleIncSIR, MultiplexDSIR):
    def __init__(self, config=None, **kwargs):
        MultiplexDSIR.__init__(self, config=config, **kwargs)
        SimpleIncSIR.__init__(self, config=config, **kwargs)


class WeightedMultiplexIncSIR(SimpleIncSIR, WeightedMultiplexDSIR):
    def __init__(self, config=None, **kwargs):
        WeightedMultiplexDSIR.__init__(self, config=config, **kwargs)
        SimpleIncSIR.__init__(self, config=config, **kwargs)


def IncSIR(config=None, **kwargs):
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexIncSIR(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedIncSIR(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexIncSIR(config=config, **kwargs)
    else:
        return SimpleIncSIR(config=config, **kwargs)
