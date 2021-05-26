import networkx as nx
import numpy as np

from .transform import NetworkTransform
from dynalearn.networks import Network, MultiplexNetwork
from dynalearn.util import (
    get_node_attr,
    get_edge_attr,
    set_node_attr,
    set_edge_attr,
)


class ThresholdNetworkTransform(NetworkTransform):
    def __init__(self, config=None, **kwargs):
        NetworkTransform.__init__(self, config=config, **kwargs)
        self.threshold = self.config.threshold
        self.collapse = self.config.collapse

    def _transform_network_(self, g):
        if isinstance(g, MultiplexNetwork) and self.collapse:
            g = g.collapse()
            g = Network(self._threshold_network(g.data))
        elif isinstance(g, MultiplexNetwork) and not self.collapse:
            data = {}
            for k, v in g.data.items():
                data[k] = self._threshold_network(v)
            g = MultiplexNetwork(data)
        else:
            g = Network(self._threshold_network(g.data))
        return g

    def _threshold_network(self, g):
        edges = np.array(list(g.to_directed().edges()))
        N = g.number_of_nodes()
        W = np.zeros((N, N))
        A = np.zeros((N, N))
        node_attr = get_node_attr(g)
        edge_attr = get_edge_attr(g)
        if "weight" not in edge_attr:
            return g
        weights = edge_attr["weight"]
        W[edges[:, 0], edges[:, 1]] = weights
        for i, w in enumerate(W.T):
            index = np.argsort(w)[::-1]
            index = index[: self.threshold]
            A[index, i] = (w[index] > 0).astype("int")
        gg = nx.DiGraph()
        gg.add_nodes_from(np.arange(N))
        gg.add_edges_from(np.array(np.where(A != 0)).T)
        gg = set_node_attr(gg, node_attr)
        assert max(dict(gg.in_degree()).values()) <= self.threshold, "Wrong in-degree."
        return gg
