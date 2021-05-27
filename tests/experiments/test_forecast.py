import unittest
from templates import *


class TrueForecastMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "TrueForecastMetrics"

    def additional_configs(self):
        self.config = ExperimentConfig.test(config="continuous")
        self.config.networks.num_nodes = 10
        self.config.train_details.num_samples = 10
        self.config.train_details.num_networks = 1


class GNNForecastMetricsTest(TrueForecastMetricsTest):
    @property
    def name(self):
        return "GNNForecastMetrics"


class VARForecastMetricsTest(TrueForecastMetricsTest):
    @property
    def name(self):
        return "VARForecastMetrics"
