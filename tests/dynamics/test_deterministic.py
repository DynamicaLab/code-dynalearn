import networkx as nx
import numpy as np
import torch
import time
import unittest

from dynalearn.dynamics import DSIS, DSIR
from dynalearn.config import DynamicsConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class DSIRTest(unittest.TestCase):
    def get_model(self):
        self._num_states = 3
        return DSIR(DynamicsConfig.dsir())

    @property
    def num_states(self):
        if self._num_states is not None:
            return self._num_states
        else:
            raise ValueError("`num_states` is not defined.")

    def setUp(self):
        self.num_nodes = 100
        self.model = self.get_model()
        self.network = get_network(NetworkConfig.barabasialbert(self.num_nodes, 2))
        self.model.network = self.network.generate(int(time.time()))

    def test_change_network(self):
        self.assertEqual(self.model.num_nodes, self.num_nodes)
        self.model.network = self.network.generate(int(time.time()))
        self.assertEqual(self.model.num_nodes, self.num_nodes)

    def test_predict(self):
        x = self.model.initial_state()
        T = 10
        for t in range(T):
            x = self.model.predict(x)
            self.assertFalse(np.any(x == np.nan))
            self.assertEqual(x.shape, (self.num_nodes, self.num_states))
            np.testing.assert_array_almost_equal(x.sum(-1), np.ones(self.num_nodes))
            print(x.sum(-1))

    def test_sample(self):
        x = self.model.initial_state()
        y = self.model.sample(x)
        self.assertTrue(np.any(x != y))

    def test_initialstate(self):
        x = self.model.initial_state()
        self.assertEqual(x.shape, (self.num_nodes, self.num_states))
