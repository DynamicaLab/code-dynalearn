import numpy as np
import unittest
from dynalearn.config import ExperimentConfig, MetricsConfig
from dynalearn.experiments import Experiment
from dynalearn.experiments.metrics import TruePEMFMetrics, GNNPEMFMetrics


class MeanfieldMetricsTest(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment(ExperimentConfig.test(), verbose=0)

        self.t_metrics = TruePEMFMetrics(MetricsConfig.test())
        self.m_metrics = GNNPEMFMetrics(MetricsConfig.test())

        self.t_metrics.initialize(self.experiment)
        self.m_metrics.initialize(self.experiment)

    def test_fixed_point(self):
        t_fp = self.t_metrics._fixed_point_()
        self.assertFalse(np.any(np.isnan(t_fp)))
        self.assertAlmostEqual(1.0, t_fp.sum())
        m_fp = self.m_metrics._fixed_point_()
        self.assertFalse(np.any(np.isnan(m_fp)))
        self.assertAlmostEqual(1.0, m_fp.sum())

    def test_all_fixed_points(self):
        self.t_metrics._all_fixed_points_()
        self.m_metrics._all_fixed_points_()
