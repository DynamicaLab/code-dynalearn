import unittest
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from dynalearn.networks import MultiplexNetwork


class MultiplexNetworkTest(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.p = 0.5
        self.layers = ["a", "b", "c"]
        self.labels = ["a", "b", "c"]
        self.network = MultiplexNetwork()

    def _generate_network_(self, layer):
        g = nx.gnp_random_graph(self.n, self.p)
        g = nx.to_directed(g)
        n = g.number_of_nodes()
        m = g.number_of_edges()
        if self.node_attr is None:
            self.node_attr = {k: np.random.randn(n) for k in self.labels}
        self.edge_attr[layer] = {k: np.random.randn(m) for k in self.labels}

        for n in g.nodes():
            for k in self.labels:
                g.nodes[n][k] = self.node_attr[k][n]

        for i, (u, v) in enumerate(g.edges()):
            for k in self.labels:
                g.edges[u, v][k] = self.edge_attr[layer][k][i]

        return g

    def _generate_multiplex_(self):
        self.node_attr = None
        self.edge_attr = {}
        return {l: self._generate_network_(l) for l in self.layers}

    def test_set_data(self):
        self.network.data = self._generate_multiplex_()

    def test_get_node_data(self):
        self.network.data = self._generate_multiplex_()
        node_data = self.network.get_node_data()
        ref_node_data = np.concatenate(
            [self.node_attr[k].reshape(-1, 1) for k in self.labels], axis=-1
        )
        np.testing.assert_array_equal(ref_node_data, node_data)

    def test_get_edge_data(self):
        self.network.data = self._generate_multiplex_()
        edge_data = self.network.get_edge_data()
        for l in self.layers:
            ref_edge_data = np.concatenate(
                [self.network.edge_attr[l][k].reshape(-1, 1) for k in self.labels],
                axis=-1,
            )
            np.testing.assert_array_equal(ref_edge_data, edge_data[l])


if __name__ == "__main__":
    unittest.main()
