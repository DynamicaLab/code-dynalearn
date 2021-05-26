import unittest
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")

from templates import TemplateConfigTest
from dynalearn.config.util import WeightConfig


class UniformWeightConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = WeightConfig.uniform()
        self.attributes = ["low", "high"]
        self.name = "UniformWeightGenerator"
