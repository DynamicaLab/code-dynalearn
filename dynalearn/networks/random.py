import networkx as nx
import numpy as np

from abc import abstractmethod
from dynalearn.config import Config
from .network import Network, MultiplexNetwork
from .generator import NetworkGenerator
from .transform import NetworkTransformList
from .weight import EmptyWeightGenerator


class RandomNetworkGenerator(NetworkGenerator):
    def __init__(self, config=None, weights=None, transforms=[], **kwargs):
        config = config or Config(**kwargs)
        NetworkGenerator.__init__(self, config)
        self.weights = weights or EmptyWeightGenerator()
        if isinstance(self.weights, EmptyWeightGenerator):
            self.is_weighted = False
        else:
            self.is_weighted = True
        self.transforms = NetworkTransformList(transforms)

    def generate(self, seed=None):
        if seed is None:
            seed = np.random.randint(2 ** 31)
        if self.layers is not None:
            g = {}
            for l in self.layers:
                g[l] = self.network(seed)
                if self.weights is not None:
                    g[l] = self.weights(g[l])
                g[l] = self.transforms(g[l])
            return MultiplexNetwork(data=g)

        else:
            g = self.network(seed)
            if self.weights is not None:
                g = self.weights(g)
            g = self.transforms(g)
            return Network(data=g)


class GNPNetworkGenerator(RandomNetworkGenerator):
    def network(self, seed=None):
        return nx.gnp_random_graph(self.num_nodes, self.config.p, seed=seed)


class GNMNetworkGenerator(RandomNetworkGenerator):
    def network(self, seed=None):
        return nx.gnm_random_graph(self.num_nodes, self.config.m, seed=seed)


class BANetworkGenerator(RandomNetworkGenerator):
    def network(self, seed=None):
        return nx.barabasi_albert_graph(self.num_nodes, self.config.m, seed)


class ConfigurationNetworkGenerator(RandomNetworkGenerator):
    def __init__(self, config=None, weights=None, **kwargs):
        config = config or Config(**kwargs)
        RandomNetworkGenerator.__init__(self, config, weights=weights, **kwargs)
        self.p_k = config.p_k
        if "maxiter" in config.__dict__:
            self.maxiter = config.maxiter
        else:
            self.maxiter = 100

    def network(self, seed=None):
        if "maxiter" in self.config.__dict__:
            maxiter = self.config.maxiter
        else:
            maxiter = 100
        it = 0
        while it < maxiter:
            seq = self.p_k.sample(self.num_nodes)
            if np.sum(seq) % 2 == 0:
                g = nx.expected_degree_graph(seq, seed=seed)
                return g
            it += 1
        raise ValueError("Invalid degree sequence.")
