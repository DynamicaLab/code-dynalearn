import networkx as nx
import numpy as np
import torch
import unittest
from dynalearn.nn.models import Propagator
from dynalearn.utilities import to_edge_index


class PropagatorTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.num_states = 4
        self.g = nx.gnp_random_graph(self.num_nodes, 0.5)
        self.edge_index = to_edge_index(self.g.to_directed())
        self.propagator = Propagator(self.num_states)

    def test_forward(self):
        x = np.random.randint(self.num_states, size=self.num_nodes)
        adj = nx.to_numpy_array(self.g)
        ref_l = np.array([adj @ (x == i) for i in range(self.num_states)])
        l = self.propagator.forward(x, self.edge_index)
        l = l.cpu().numpy()
        torch.testing.assert_allclose(l, ref_l)


if __name__ == "__main__":
    unittest.main()
