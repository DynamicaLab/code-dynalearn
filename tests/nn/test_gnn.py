import networkx as nx
import numpy as np
import torch
import unittest

from dynalearn.nn.models import GraphNeuralNetwork
from dynalearn.config import TrainableConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class GraphNeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.window_size = 2
        self.nodeattr_size = 2
        self.edgeattr_size = 0
        self.num_nodes = 10
        self.model = GraphNeuralNetwork(
            self.in_size,
            self.out_size,
            window_size=self.window_size,
            nodeattr_size=self.nodeattr_size,
            edgeattr_size=self.edgeattr_size,
            out_act="softmax",
            normalize=True,
            config=TrainableConfig.sis(),
        )
        self.network = get_network(NetworkConfig.w_ba(self.num_nodes, 2))

    def test_forward(self):
        x = torch.randn(self.num_nodes, self.in_size, self.window_size)
        g = self.network.generate()
        g.node_attr = {"na": np.random.randn(self.num_nodes, self.nodeattr_size)}
        g.edge_attr = {"ea": np.random.randn(g.number_of_edges(), self.edgeattr_size)}

        x = self.model.transformers["t_inputs"].forward(x)
        g = self.model.transformers["t_networks"].forward(g)
        y = self.model.forward(x, g).cpu().detach().numpy()
        self.assertTrue(y.shape == (self.num_nodes, self.out_size))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones((y.shape[0])))


if __name__ == "__main__":
    unittest.main()
