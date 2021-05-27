import unittest
import networkx as nx
import numpy as np

from .templates import *
from dynalearn.dynamics import GNNSEDynamics, GNNDEDynamics, GNNIncidenceDynamics
from dynalearn.config import TrainableConfig, NetworkConfig
from torch_geometric.nn.inits import ones
from dynalearn.networks.getter import get as get_network


class GNNSEDynamicsTest(unittest.TestCase):
    def get_model(self):
        self.config = TrainableConfig.sis()
        return GNNSEDynamics(self.config)

    def setUp(self):
        self.model = self.get_model()
        self.num_nodes = 5
        self.num_states = self.config.num_states
        self.lag = self.config.lag
        self.network = get_network(NetworkConfig.ba(self.num_nodes))
        self.predict_shape = (self.num_nodes, self.num_states)
        self.sample_shape = (self.num_nodes,)

    def test_predict(self):
        for i in range(10):
            self.model.nn.reset_parameters()
            self.model.network = self.network.generate(0)
            x = self.model.initial_state(squeeze=False)
            y = self.model.predict(x)
            self.assertFalse(np.any(y == np.nan))
            self.assertEqual(y.shape, self.predict_shape)
            np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.num_nodes))

    def test_sample(self):
        self.model.network = self.network.generate(0)
        x = self.model.initial_state(squeeze=False)
        self.model.nn.reset_parameters()
        y = self.model.sample(x)
        self.assertEqual(y.shape, self.sample_shape)
        self.model.nn.reset_parameters()


class GNNDEDynamicsTest(GNNSEDynamicsTest):
    def get_model(self):
        self.config = TrainableConfig.dsir()
        self.config.is_weighted = True
        self.config.is_multiplex = False
        return GNNDEDynamics(self.config)

    def setUp(self):
        self.model = self.get_model()
        self.num_nodes = 5
        self.num_states = self.config.num_states
        self.lag = self.config.lag
        self.network = get_network(NetworkConfig.w_ba(self.num_nodes))
        self.predict_shape = (self.num_nodes, self.num_states)
        self.sample_shape = (self.num_nodes, self.num_states)


class GNNIncidenceDynamicsTest(GNNSEDynamicsTest):
    def get_model(self):
        self.config = TrainableConfig.incsir()
        self.config.is_weighted = True
        self.config.is_multiplex = False
        return GNNIncidenceDynamics(self.config)

    def setUp(self):
        self.model = self.get_model()
        self.num_nodes = 5
        self.num_states = self.config.num_states
        self.lag = self.config.lag
        self.network = get_network(NetworkConfig.w_ba(self.num_nodes))
        self.predict_shape = (self.num_nodes, self.num_states)
        self.sample_shape = (self.num_nodes, self.num_states)

    def test_predict(self):
        for i in range(10):
            self.model.nn.reset_parameters()
            self.model.network = self.network.generate(0)
            x = self.model.initial_state(squeeze=False)
            y = self.model.predict(x)
            self.assertFalse(np.any(y == np.nan))
            self.assertEqual(y.shape, self.predict_shape)


if __name__ == "__main__":
    unittest.main()
