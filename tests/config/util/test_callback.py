import unittest
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")

from dynalearn.config.util import CallbackConfig
from templates import TemplateConfigTest


class CallbackConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = CallbackConfig.default()
        self.attributes = ["step_size", "gamma", "path_to_best"]
