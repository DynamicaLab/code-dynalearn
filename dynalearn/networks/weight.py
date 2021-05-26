import networkx as nx
import numpy as np

from abc import abstractmethod, ABC
from dynalearn.config import Config


class WeightGenerator(ABC):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

    @abstractmethod
    def generate(self, u, v):
        raise NotImplemented

    def setUp(self, g):
        return

    def __call__(self, g):
        _g = g.copy()
        self.setUp(g)
        for u, v in g.edges():
            _g.edges[u, v]["weight"] = self.generate(u, v)
        return _g


class EmptyWeightGenerator(WeightGenerator):
    def generate(self, u, v):
        return

    def __call__(self, g):
        return g


class UniformWeightGenerator(WeightGenerator):
    def __init__(self, config=None, **kwargs):
        WeightGenerator.__init__(self, config=config, **kwargs)
        if "low" in config.__dict__:
            self.low = config.low
        else:
            self.low = 0

        if "high" in config.__dict__:
            self.high = config.high
        else:
            self.high = 1

    def generate(self, u, v):
        r = np.random.rand()
        return (1 - r) * self.low + r * self.high


class LogUniformWeightGenerator(UniformWeightGenerator):
    def generate(self, u, v):
        r = np.random.rand()
        return np.exp((1 - r) * np.log(self.low) + r * np.log(self.high))


class NormalWeightGenerator(WeightGenerator):
    def __init__(self, config=None, **kwargs):
        WeightGenerator.__init__(self, config=config, **kwargs)

        if "mean" in config.__dict__:
            self.mean = config.mean
        else:
            self.mean = 0

        if "std" in config.__dict__:
            self.std = config.std
        else:
            self.std = 1

    def generate(self, u, v):
        r = np.random.randn()
        return r * self.std + self.mean


class LogNormalWeightGenerator(NormalWeightGenerator):
    def generate(self, u, v):
        r = np.random.randn()
        return np.exp(r * np.log(self.std) + np.log(self.mean))


class DegreeWeightGenerator(WeightGenerator):
    def __init__(self, config=None, **kwargs):
        WeightGenerator.__init__(self, config=config, **kwargs)

        if "normalized" in config.__dict__:
            self.normalized = config.normalized
            if "mean" in config.__dict__:
                self.mean = config.mean
            else:
                self.mean = 0

                if "std" in config.__dict__:
                    self.std = config.std
                else:
                    self.std = 1
        else:
            self.normalized = False

    def setUp(self, g):
        d = g.degree()
        m = 2 * self.number_of_edges()
        self.degree_weight = {(u, v): d[u] * d[v] / m for u, v in g.edges()}
        if self.normalized:
            mean = np.mean(list(self.degree_weight.values()))
            std = np.std(list(self.degree_weight.values()))
            for k, v in self.degree_weight:
                r = (self.degree_weight[k] - mean) / std
                self.degree_weight[k] = r * self.std + self.mean

    def generate(self, u, v):
        return self.degree_weight[(u, v)]


class BetweennessWeightGenerator(WeightGenerator):
    def __init__(self, config=None, **kwargs):
        WeightGenerator.__init__(self, config=config, **kwargs)

        if "normalized" in config.__dict__:
            self.normalized = config.normalized
            if "mean" in config.__dict__:
                self.mean = config.mean
            else:
                self.mean = 0

                if "std" in config.__dict__:
                    self.std = config.std
                else:
                    self.std = 1
        else:
            self.normalized = False

    def setUp(self, g):
        self.betweenness = nx.edge_betweenness_centrality(g, normalized=False)
        if self.normalized:
            mean = np.mean(list(self.betweenness.values()))
            std = np.std(list(self.betweenness.values()))
            for k, v in self.betweenness:
                r = (self.betweenness[k] - mean) / std
                self.betweenness[k] = r * self.std + self.mean

    def generate(self, u, v):
        return self.betweenness[(u, v)]
