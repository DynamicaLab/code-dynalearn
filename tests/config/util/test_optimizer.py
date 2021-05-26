import unittest
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")

from dynalearn.config.util import OptimizerConfig
from templates import TemplateConfigTest


class OptimizerConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = OptimizerConfig.default()
        self.attributes = ["lr", "weight_decay", "betas", "eps", "amsgrad"]
        self.name = "RAdam"
