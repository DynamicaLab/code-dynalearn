import unittest

from .templates import TemplateConfigTest
from dynalearn.config.util import (
    AttentionConfig,
    ForecastConfig,
    LTPConfig,
    PredictionConfig,
    StationaryConfig,
    StatisticsConfig,
)


class AttentionConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = AttentionConfig.default()
        self.attributes = ["max_num_points"]


class ForecastConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ForecastConfig.default()
        self.attributes = ["num_steps"]


class LTPConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = LTPConfig.default()
        self.attributes = ["max_num_sample", "max_num_points"]


class PredictionConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = PredictionConfig.default()
        self.attributes = ["max_num_points"]


class StoContStationaryConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = StationaryConfig.sis()
        self.attributes = [
            "adaptive",
            "num_nodes",
            "init_param",
            "sampler",
            "burn",
            "T",
            "tol",
            "num_samples",
            "statistics",
            "parameters",
        ]


class MetapopStationaryConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = StationaryConfig.dsir()
        self.attributes = [
            "adaptive",
            "num_nodes",
            "init_param",
            "sampler",
            "initial_burn",
            "init_epsilon",
            "mid_burn",
            "tol",
            "maxiter",
            "num_samples",
            "statistics",
            "parameters",
        ]


class StatisticsConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = StatisticsConfig.default()
        self.attributes = ["maxlag", "max_num_points"]
