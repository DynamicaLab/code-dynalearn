import numpy as np
import time

from dynalearn.config import NetworkConfig
from dynalearn.networks.getter import get as get_network

NUM_SAMPLES = 10


class StoContTemplateTest:
    def get_model(self):
        raise NotImplemented()

    @property
    def num_states(self):
        if self._num_states is not None:
            return self._num_states
        else:
            raise ValueError("`num_states` is not defined.")

    def setUp(self):
        self.num_nodes = 100
        self.model = self.get_model()
        self.network = get_network(NetworkConfig.ba(self.num_nodes, 2))
        self.model.network = self.network.generate(int(time.time()))

    def test_change_network(self):
        self.assertEqual(self.model.num_nodes, self.num_nodes)
        self.model.network = self.network.generate(int(time.time()))
        self.assertEqual(self.model.num_nodes, self.num_nodes)

    def test_predict(self):
        x = np.random.randint(self.num_states, size=self.num_nodes)
        y = self.model.predict(x)
        self.assertFalse(np.any(y == np.nan))
        self.assertEqual(y.shape, (self.num_nodes, self.num_states))
        np.testing.assert_array_almost_equal(y.sum(-1), np.ones(self.num_nodes))

    def test_sample(self):
        success = 0
        x = np.zeros(self.num_nodes)
        self.assertTrue(np.all(x == self.model.sample(x)))
        for i in range(NUM_SAMPLES):
            x = np.random.randint(self.num_states, size=self.num_nodes)
            y = self.model.sample(x)
            success += int(np.any(x != y))
        self.assertGreater(success, 0)

    def test_initialstate(self):
        x = self.model.initial_state()
        self.assertEqual(x.shape, (self.num_nodes,))
