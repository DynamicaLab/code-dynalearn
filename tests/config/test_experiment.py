import unittest
import warnings
import os

warnings.filterwarnings("ignore")

from dynalearn.config import ExperimentConfig
from templates import TemplateConfigTest

EXP_ATTRIBUTES = [
    "path_to_data",
    "path_to_best",
    "path_to_summary",
    "dynamics",
    "networks",
    "model",
    "metrics",
    "train_metrics",
    "callbacks",
    "seed",
]


class ExperimentConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.default("test-exp", "sis", "ba")
        self.attributes = EXP_ATTRIBUTES
        self.name = "test-exp"

    def tearDown(self):
        os.removedirs(f"./{self.name}")


class COVIDExperimentConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.covid("test-covid-exp")
        self.attributes = EXP_ATTRIBUTES
        self.name = "test-covid-exp"

    def tearDown(self):
        os.removedirs(f"./{self.name}")
