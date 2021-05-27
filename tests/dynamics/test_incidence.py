import networkx as nx
import numpy as np
import torch
import time
import unittest

from .templates import *
from dynalearn.dynamics import IncSIR
from dynalearn.config import DynamicsConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class IncidenceSIRTest(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.model = IncSIR(DynamicsConfig.incsir())
        self.num_states = self.model.num_states
        self.lag = self.model.lag
        self.network = get_network(NetworkConfig.ba(self.num_nodes, 2))
        self.model.network = self.network.generate(int(time.time()))

    def test_change_network(self):
        self.assertEqual(self.model.num_nodes, self.num_nodes)
        self.model.network = self.network.generate(int(time.time()))
        self.assertEqual(self.model.num_nodes, self.num_nodes)

    def test_predict(self):
        x = np.random.poisson(5, size=(self.num_nodes, 1))
        T = 10
        for t in range(T):
            x = self.model.predict(x)
            self.assertFalse(np.any(x == np.nan))
            self.assertEqual(x.shape, (self.num_nodes, self.num_states))
            np.testing.assert_array_almost_equal(
                self.model.latent_state.sum(-1), np.ones(self.num_nodes)
            )
            self.assertTrue(np.all(self.model.latent_state >= 0))
            self.assertTrue(np.all(self.model.latent_state <= 1))

    def test_sample(self):
        x = self.model.initial_state()
        y = self.model.sample(x)
        self.assertTrue(np.any(x != y))

    def test_initialstate(self):
        x = self.model.initial_state(squeeze=False)
        self.assertEqual(x.shape, (self.num_nodes, self.num_states, self.lag))
