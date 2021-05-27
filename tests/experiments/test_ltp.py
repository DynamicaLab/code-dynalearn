import unittest
from templates import *


class TrueLTPMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "TrueLTPMetrics"

    def check_data(self, data):
        ltp = data["ltp"]
        np.testing.assert_array_almost_equal(ltp.sum(-1), np.ones(ltp.shape[0]))


class GNNLTPMetricsTest(TrueLTPMetricsTest):
    @property
    def name(self):
        return "GNNLTPMetrics"
