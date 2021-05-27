import unittest

from dynalearn.config.util import WeightConfig
from .templates import TemplateConfigTest


class UniformWeightConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = WeightConfig.uniform()
        self.attributes = ["low", "high"]
        self.name = "UniformWeightGenerator"
