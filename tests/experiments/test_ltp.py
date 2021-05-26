import networkx as nx
import numpy as np

from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment
from unittest import TestCase


class TestLTPMetrics(TestCase):
    def setUp(self):
        self.n = 100
        self.num_samples = 5
        self.num_networks = 2
        self.config = ExperimentConfig.test()

        self.config.train_details.num_samples = self.num_samples
        self.config.train_details.num_networks = self.num_networks
        self.config.metrics.names = ["TrueLTPMetrics", "GNNLTPMetrics"]
        self.config.networks.num_nodes = self.n

        self.experiment = Experiment(self.config)
        self.experiment.generate_data()
        self.experiment.compute_metrics()

    def test_compute(self):
        summ = self.experiment.metrics["TrueLTPMetrics"].data["summaries"]
        ltp = self.experiment.metrics["TrueLTPMetrics"].data["ltp"]

        # susceptible
        s_summ = summ[summ[:, 0] == 0]
        s_ltp = ltp[summ[:, 0] == 0]
        p = (1 - self.experiment.dynamics.infection) ** s_summ[:, 2]
        ref_s_ltp = np.array([p, 1 - p]).T
        np.testing.assert_array_almost_equal(ref_s_ltp, s_ltp)

        # infected
        i_summ = summ[summ[:, 0] == 1]
        i_ltp = ltp[summ[:, 0] == 1]
        p = self.experiment.dynamics.recovery * np.ones(i_summ.shape[0])
        ref_i_ltp = np.array([p, 1 - p]).T
        np.testing.assert_array_almost_equal(ref_i_ltp, i_ltp)

        summ = self.experiment.metrics["GNNLTPMetrics"].data["summaries"]
        ltp = self.experiment.metrics["GNNLTPMetrics"].data["ltp"]
        self.assertTrue(ltp.shape == (summ.shape[0], summ.shape[1] - 1))
        self.assertTrue(np.all(ltp <= 1))
        self.assertTrue(np.all(ltp >= 0))
