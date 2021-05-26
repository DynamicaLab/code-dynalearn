import numpy as np

from .base import (
    IncidenceEpidemics,
    WeightedIncidenceEpidemics,
    MultiplexIncidenceEpidemics,
    WeightedMultiplexIncidenceEpidemics,
)
from dynalearn.config import Config

EPSILON = 0.0


class SimpleIncidenceSIR(IncidenceEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        IncidenceEpidemics.__init__(self, config)
        self.infection_prob = config.infection_prob
        self.infection_type = config.infection_type
        self.recovery_prob = config.recovery_prob
        self._num_infected = None
        self._num_recovered = None

    @property
    def num_infected(self):
        assert self._num_infected is not None
        return self._num_infected

    @num_infected.setter
    def num_infected(self, num_infected):
        self._num_infected = num_infected

    @property
    def num_recovered(self):
        assert self._num_recovered is not None
        return self._num_recovered

    @num_recovered.setter
    def num_recovered(self, num_recovered):
        self._num_recovered = num_recovered

    def predict(self, x):
        if x.ndim == 3:
            x = x[:, 0, -1]
        if self._num_infected is None:
            self.num_infected = x * 1
        if self._num_recovered is None:
            self.num_recovered = np.zeros(x.shape)
        assert np.all(
            self.num_infected + self.num_recovered < self.population
        ), f"Invalid sequence, i = {self.num_infected.sum()}, r = {self.num_recovered.sum()}, n = {self.population.sum()}"
        infection = self.propagate(self.infection_rate(x).squeeze())
        recovery = self.recovery_rate(x).squeeze()
        i = self.num_infected * 1
        r = self.num_recovered * 1
        s = self.population - i - r
        p = s * infection
        self.num_infected = i + s * infection - i * recovery
        self.num_recovered = r + i * recovery
        return p.reshape(-1, 1)

    def infection_rate(self, x):
        if self.infection_type == 1:
            return 1 - (1 - self.infection_prob) ** self.num_infected
        elif self.infection_type == 2:
            print(self.infection_prob, self.num_infected, self.population)
            return 1 - (1 - self.infection_prob / self.population) ** self.num_infected

    def recovery_rate(self, x):
        return self.recovery_prob * np.ones(x.shape[0])


class WeightedIncidenceSIR(SimpleIncidenceSIR, WeightedIncidenceEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleIncidenceSIR.__init__(self, config=config, **kwargs)
        WeightedIncidenceEpidemics.__init__(self, config)


class MultiplexIncidenceSIR(SimpleIncidenceSIR, MultiplexIncidenceEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleIncidenceSIR.__init__(self, config=config, **kwargs)
        MultiplexIncidenceEpidemics.__init__(self, config)


class WeightedMultiplexIncidenceSIR(
    SimpleIncidenceSIR, WeightedMultiplexIncidenceEpidemics
):
    def __init__(self, config=None, **kwargs):
        SimpleIncidenceSIR.__init__(self, config=config, **kwargs)
        WeightedMultiplexIncidenceEpidemics.__init__(self, config)


def IncidenceSIR(config=None, **kwargs):
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexIncidenceSIR(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedIncidenceSIR(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexIncidenceSIR(config=config, **kwargs)
    else:
        return SimpleIncidenceSIR(config=config, **kwargs)
