import numpy as np

from random import sample


class NetworkTransform:
    def __init__(self, config):
        self.config = config

    def __call__(self, g):
        return g


class NetworkTransformList:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, g):
        for t in self.transforms:
            g = t(g)
        return g


class SparcifierTransform(NetworkTransform):
    def __call__(self, g):
        _g = g.copy()
        for i in range(self.config.maxiter):
            if self.config.p == -1:
                p = np.random.rand()
                p = 1 - np.log((1 - p) + np.exp(1) * p)
            else:
                p = self.config.p
            num_edges = np.random.binomial(_g.number_of_edges(), p)
            removed_edges = sample(_g.edges, num_edges)
            _g.remove_edges_from(removed_edges)
            if _g.number_of_edges() == 0:
                _g = g.copy()
            else:
                break
        return _g
