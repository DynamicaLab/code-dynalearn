import unittest
from dynalearn.config import ExperimentConfig, MetricsConfig
from dynalearn.experiments import Experiment
from dynalearn.experiments.metrics import TruePSSMetrics, GNNPSSMetrics


class StationaryStateMetricsTest(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(ExperimentConfig.test(), verbose=0)

        self.t_metrics = TruePSSMetrics(MetricsConfig.test())
        self.m_metrics = GNNPSSMetrics(MetricsConfig.test())

        self.t_metrics.initialize(self.experiment)
        self.m_metrics.initialize(self.experiment)

    def test_stationary_state(self):
        self.t_metrics._all_stationary_states_()
        # self.m_metrics._all_stationary_states_()
