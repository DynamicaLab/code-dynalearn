import networkx as nx
import numpy as np
import unittest

from dynalearn.datasets.weights import (
    Weight,
    DegreeWeight,
    StrengthWeight,
    DiscreteStateWeight,
    DiscreteCompoundStateWeight,
    ContinuousStateWeight,
    ContinuousCompoundStateWeight,
    StrengthContinuousStateWeight,
    StrengthContinuousCompoundStateWeight,
)
from dynalearn.utilities import set_edge_attr, get_node_strength


class WeightTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.num_states = 3
        self.size = 10

        g = nx.barabasi_albert_graph(self.num_nodes, 2)
        w = np.random.randn(2 * g.number_of_edges())
        self.g = set_edge_attr(g, {"weight": w})
        self.d_x = np.random.randint(
            self.num_states, size=(self.size, self.num_nodes, 1)
        )
        self.d_templates = [
            Weight,
            DegreeWeight,
            StrengthWeight,
            DiscreteStateWeight,
            DiscreteCompoundStateWeight,
        ]
        self.c_x = (
            np.random.randn(self.size, self.num_nodes, self.num_states, 1) * 100 + 500
        )
        self.c_templates = [
            Weight,
            DegreeWeight,
            StrengthWeight,
            ContinuousStateWeight,
            ContinuousCompoundStateWeight,
            StrengthContinuousStateWeight,
            StrengthContinuousCompoundStateWeight,
        ]

    def test_get_features(self):
        for t in self.d_templates:
            weight = t()
            weight.num_states = self.num_states
            weight.window_size = 1
            weight._get_features_(self.g, self.d_x)

        for t in self.c_templates:
            weight = t()
            weight._get_features_(self.g, self.c_x)

    def test_get_weights(self):
        for t in self.d_templates:
            weight = t()
            weight.num_states = self.num_states
            weight.window_size = 1
            weight._get_features_(self.g, self.d_x)
            w = weight._get_weights_(self.g, self.d_x)
            self.assertTrue(np.all(w > 0))

        for t in self.c_templates:
            weight = t()
            weight._get_features_(self.g, self.c_x)
            w = weight._get_weights_(self.g, self.c_x)
            self.assertTrue(np.all(w > 0))


if __name__ == "__main__":
    unittest.main()
