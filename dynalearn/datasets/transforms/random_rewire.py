import networkx as nx

from dynalearn.datasets.transforms import NetworkTransform


class RandomRewireNetworkTransform(NetworkTransform):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.rewire = config.rewire
        Transform.__init__(self, config, **kwargs)

    def _transform_network_(self, g):
        g = g.data
        num_edges = g.number_of_edges()
        n = np.random.binomial(num_edges, self.rewire)
        return nx.double_edge_swap(g.copy(), nswap=n, seed=np.random.randint(2 ** 31))
