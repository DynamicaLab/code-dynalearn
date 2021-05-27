import unittest
from templates import *
from dynalearn.config.util import StationaryConfig


class TrueERSSMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "TrueERSSMetrics"


class GNNERSSMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "GNNERSSMetrics"
