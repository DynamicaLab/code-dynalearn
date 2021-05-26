import networkx as nx
import numpy as np

from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment
from unittest import TestCase


class TestpredictionMetrics(TestCase):
    def setUp(self):
        self.n = 100
        self.num_samples = 5
        self.num_networks = 2
        self.config = ExperimentConfig.test()

        self.config.train_details.num_samples = self.num_samples
        self.config.train_details.num_networks = self.num_networks
        self.config.metrics.names = ["TruePredictionMetrics", "GNNPredictionMetrics"]
        self.config.metrics.max_num_points = 1e3
        self.config.networks.num_nodes = self.n


        self.experiment = Experiment(self.config)
        self.experiment.generate_data()
        self.experiment.compute_metrics()

    def test_compute(self):
        pred = self.experiment.metrics["TruePredictionMetrics"].data["pred"]
        self.assertTrue(np.all(pred <= 1))
        self.assertTrue(np.all(pred >= 0))
        np.testing.assert_array_almost_equal(pred.sum(-1), np.ones(pred.shape[0]))

        pred = self.experiment.metrics["GNNPredictionMetrics"].data["pred"]
        self.assertTrue(np.all(pred <= 1))
        self.assertTrue(np.all(pred >= 0))
        np.testing.assert_array_almost_equal(pred.sum(-1), np.ones(pred.shape[0]))
