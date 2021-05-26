import networkx as nx
import numpy as np
import torch
import unittest
from dynalearn.nn.transformers import (
    InputNormalizer,
    TargetNormalizer,
    NodeNormalizer,
    EdgeNormalizer,
    NetworkNormalizer,
)
from dynalearn.utilities import get_node_attr, get_edge_attr
from dynalearn.config import NetworkConfig
from dynalearn.networks.getter import get as get_network


class InputNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.num_states = 4
        self.mean = 5 * torch.ones(torch.Size([1, self.num_states, 1]))
        self.var = 1 * torch.ones(torch.Size([1, self.num_states, 1]))
        self.normalizer = InputNormalizer(self.num_states)
        self.normalizer.inputs_mean = self.mean
        self.normalizer.inputs_var = self.var

    def test_forwardbackward(self):
        n = 10
        ws = 2

        x = self.mean + torch.randn(n, self.num_states, ws) * self.var ** (0.5)
        y = self.normalizer.forward(x)
        z = self.normalizer.backward(y)
        x = x.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        np.testing.assert_array_almost_equal(x, z)


class TargetNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.num_states = 4
        self.mean = 5 * torch.ones(torch.Size([1, self.num_states]))
        self.var = 1 * torch.ones(torch.Size([1, self.num_states]))
        self.normalizer = TargetNormalizer(self.num_states)
        self.normalizer.targets_mean = self.mean
        self.normalizer.targets_var = self.var

    def test_forwardbackward(self):
        n = 10

        x = self.mean + torch.randn(n, self.num_states) * self.var ** (0.5)
        y = self.normalizer.forward(x)
        z = self.normalizer.backward(y)
        x = x.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        np.testing.assert_array_almost_equal(x, z)


class NodeNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.num_attr = 2
        self.mean = 5 * torch.ones(torch.Size([1, self.num_attr]))
        self.var = 1 * torch.ones(torch.Size([1, self.num_attr]))
        self.normalizer = NodeNormalizer(self.num_attr)
        self.normalizer.nodeattr_mean = self.mean
        self.normalizer.nodeattr_var = self.var

    def test_forwardbackward(self):
        g = nx.gnp_random_graph(10, 0.5)
        n = g.number_of_nodes()
        m = 2 * g.number_of_edges()
        for u in g.nodes():
            for i in range(self.num_attr):
                m = float(self.mean[0, i].squeeze())
                s = float(self.var[0, i].squeeze()) ** (0.5)
                g.nodes[u][f"attr{i}"] = m + np.random.randn() * s
        attr = get_node_attr(g, to_data=True)
        norm_attr = self.normalizer.forward(attr)
        notnorm_attr = self.normalizer.backward(norm_attr).cpu().detach().numpy()

        self.assertFalse(np.any(attr == notnorm_attr))
        np.testing.assert_array_almost_equal(attr, notnorm_attr)


class EdgeNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.num_attr = 2
        self.mean = 5 * torch.ones(torch.Size([1, self.num_attr]))
        self.var = 1 * torch.ones(torch.Size([1, self.num_attr]))
        self.normalizer = EdgeNormalizer(self.num_attr)
        self.normalizer.edgeattr_mean = self.mean
        self.normalizer.edgeattr_var = self.var

    def test_forwardbackward(self):
        g = nx.gnp_random_graph(10, 0.5)
        n = g.number_of_nodes()
        m = 2 * g.number_of_edges()
        for u, v in g.edges():
            for i in range(self.num_attr):
                m = float(self.mean[0, i].squeeze())
                s = float(self.var[0, i].squeeze()) ** (0.5)
                g.edges[u, v][f"attr{i}"] = m + np.random.randn() * s
        attr = get_edge_attr(g, to_data=True)
        norm_attr = self.normalizer.forward(attr)
        notnorm_attr = self.normalizer.backward(norm_attr).cpu().detach().numpy()

        self.assertFalse(np.any(attr == notnorm_attr))
        np.testing.assert_array_almost_equal(attr, notnorm_attr)


class NetworkNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.node_size = 0
        self.node_mean = torch.zeros(torch.Size([1, self.node_size]))
        self.node_var = torch.ones(torch.Size([1, self.node_size]))

        self.edge_size = 0
        self.edge_mean = torch.zeros(torch.Size([1, self.edge_size]))
        self.edge_var = torch.ones(torch.Size([1, self.edge_size]))

        self.normalizer = NetworkNormalizer(self.node_size, self.edge_size)
        self.normalizer.nodeattr_mean = self.node_mean
        self.normalizer.nodeattr_var = self.node_var
        self.normalizer.edgeattr_mean = self.edge_mean
        self.normalizer.edgeattr_var = self.edge_var

        self.num_nodes = 10
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes, 2))

    def test_forward(self):
        g = self.network.generate()
        N, M = g.number_of_nodes(), g.number_of_edges()
        if self.node_size > 0:
            node_attr = {}
            for i in range(self.node_size):
                m = float(self.node_mean[0, i].squeeze())
                s = float(self.node_var[0, i].squeeze()) ** (0.5)
                node_attr[f"attr{i}"] = m + np.random.randn(g.number_of_nodes()) * s
            g.node_attr = node_attr
        if self.edge_size > 0:
            edge_attr = {}
            for i in range(self.edge_size):
                m = float(self.edge_mean[0, i].squeeze())
                s = float(self.edge_var[0, i].squeeze()) ** (0.5)
                edge_attr[f"attr{i}"] = m + np.random.randn(g.number_of_edges()) * s
            g.edge_attr = edge_attr
        _g = self.normalizer.forward(g)
        self.assertTrue(isinstance(_g, tuple))
        self.assertTrue(len(_g) == 3)
        self.assertTrue(_g[0].shape == (2, M))
        if self.edge_size > 0:
            self.assertTrue(_g[1].shape == (M, self.edge_size))
        else:
            self.assertTrue(_g[1] is None)
        if self.node_size > 0:
            self.assertTrue(_g[2].shape == (N, self.node_size))
        else:
            self.assertTrue(_g[2] is None)


if __name__ == "__main__":
    unittest.main()
