import unittest
from templates import *


class PredictionMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "PredictionMetrics"

    def check_data(self, data):
        pred = data["pred"]
        self.assertTrue(np.all(pred <= 1))
        self.assertTrue(np.all(pred >= 0))
        np.testing.assert_array_almost_equal(pred.sum(-1), np.ones(pred.shape[0]))
