import unittest
from .templates import *


class TestStatisticsMetrics(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "StatisticsMetrics"
