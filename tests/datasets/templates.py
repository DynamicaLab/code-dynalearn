import networkx as nx
import numpy as np
import os
import torch
import unittest
import warnings

warnings.filterwarnings("ignore")

from dynalearn.config import ExperimentConfig
from dynalearn.networks import Network
from dynalearn.experiments import Experiment
from dynalearn.util import set_edge_attr


class TemplateDatasetTest:
    def get_config(self):
        raise NotImplemented()

    def get_input_shape(self):
        raise NotImplemented()

    def setUp(self):
        self.config = self.get_config()  # ExperimentConfig.test(config="continuous")
        self.num_networks = 2
        self.num_samples = 10
        self.num_nodes = 10
        self.p = 10
        self.batch_size = 5
        self.config.train_details.num_networks = self.num_networks
        self.config.train_details.num_samples = self.num_samples
        self.config.networks.num_nodes = self.num_nodes
        self.config.networks.p = self.p
        self.exp = Experiment(self.config, verbose=0)
        self.dataset = self.exp.dataset
        self.dataset.setup(self.exp)
        self.lag = self.exp.model.lag
        self.num_states = self.exp.model.num_states
        data = self.dataset._generate_data_(self.exp.train_details)
        self.dataset.data = data

    def tearDown(self):
        os.remove(f"./{self.exp.name}")

    def test_get_indices(self):
        indices = self.dataset.indices
        ref_indices = list(range(self.num_networks * self.num_samples))
        self.assertEqual(self.num_networks * self.num_samples, len(indices))
        self.assertEqual(ref_indices, list(indices.keys()))

    def test_get_weights(self):
        weights = self.dataset.weights
        for i in range(self.num_networks):
            self.assertEqual(weights[i].data.shape, (self.num_samples, self.num_nodes))

    def test_partition(self):
        dataset = self.dataset.partition(type="random", fraction=0.5)
        for i in range(self.num_networks):
            np.testing.assert_array_equal(self.dataset.networks[i], dataset.networks[i])
            np.testing.assert_array_equal(
                self.dataset.inputs[i].data, dataset.inputs[i].data
            )
            np.testing.assert_array_equal(
                self.dataset.targets[i].data, dataset.targets[i].data
            )
            index1 = self.dataset.weights[i].data == 0.0
            index2 = dataset.weights[i].data > 0.0
            np.testing.assert_array_equal(index1, index2)
        return

    def test_next(self):
        it = iter(self.dataset)
        data = next(it)
        i = 0
        for data in self.dataset:
            i += 1
        self.assertEqual(self.num_samples * self.num_networks, i)
        (x, g), y, w = data
        self.assertEqual(self.get_input_shape(), x.shape)
        self.assertEqual((self.num_nodes, self.num_states), y.shape)
        self.assertEqual((self.num_nodes,), w.shape)

    def test_batch(self):
        batches = self.dataset.to_batch(self.batch_size)
        i = 0
        for b in batches:
            j = 0
            for bb in b:
                (x, g), y, w = bb
                self.assertEqual(type(x), torch.Tensor)
                self.assertTrue(issubclass(g.__class__, Network))
                self.assertTrue(issubclass(g.data.__class__, nx.Graph))
                self.assertEqual(type(y), torch.Tensor)
                self.assertEqual(type(w), torch.Tensor)
                j += 1
                i += 1
                pass
            self.assertEqual(self.batch_size, j)
        self.assertEqual(self.num_samples * self.num_networks, i)


class DiscreteWeightTest:
    def get_weight(self):
        raise NotImplemented()

    def get_state(self):
        return np.random.randint(self.num_states, size=(self.size, self.num_nodes, 1))

    def setUp(self):
        self.num_nodes = 10
        self.num_states = 3
        self.size = 10

        g = nx.barabasi_albert_graph(self.num_nodes, 2)
        w = np.random.randn(2 * g.number_of_edges())
        g = set_edge_attr(g, {"weight": w})
        self.g = Network(data=g)
        self.state = self.get_state()

    def test_get_features(self):
        weight = self.get_weight()
        weight.num_states = self.num_states
        weight.lag = 1
        weight._get_features_(self.g, self.state)

    def test_get_weights(self):
        weight = self.get_weight()
        weight.num_states = self.num_states
        weight.lag = 1
        weight._get_features_(self.g, self.state)
        w = weight._get_weights_(self.g, self.state)
        self.assertTrue(np.all(w > 0))


class ContinuousWeightTest:
    def get_weight(self):
        raise NotImplemented()

    def get_state(self):
        return (
            np.random.randn(self.size, self.num_nodes, self.num_states, 1) * 100 + 500
        )

    def setUp(self):
        self.num_nodes = 10
        self.num_states = 3
        self.size = 10

        g = nx.barabasi_albert_graph(self.num_nodes, 2)
        w = np.random.randn(2 * g.number_of_edges())
        g = set_edge_attr(g, {"weight": w})
        self.g = Network(data=g)
        self.state = self.get_state()

    def test_get_features(self):
        weight = self.get_weight()
        weight._get_features_(self.g, self.state)

    def test_get_weights(self):
        weight = self.get_weight()
        weight._get_features_(self.g, self.state)
        w = weight._get_weights_(self.g, self.state)
        self.assertTrue(np.all(w > 0))
