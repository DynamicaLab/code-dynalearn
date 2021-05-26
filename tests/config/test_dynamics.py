import unittest
import warnings

warnings.filterwarnings("ignore")

from dynalearn.config import DynamicsConfig
from templates import TemplateConfigTest


class SISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sis()
        self.name = "SIS"
        self.attributes = ["infection", "recovery", "init_param"]


class PlanckSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.plancksis()
        self.name = "PlanckSIS"
        self.attributes = ["temperature", "recovery", "init_param"]


class SISSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sissis()
        self.name = "AsymmetricSISSIS"
        self.attributes = [
            "infection1",
            "infection2",
            "recovery1",
            "recovery2",
            "coupling",
            "boost",
            "init_param",
        ]


class DSIRConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.dsir()
        self.name = "DSIR"
        self.attributes = [
            "infection_prob",
            "recovery_prob",
            "infection_type",
            "density",
            "init_param",
        ]


class DSIRConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.incsir()
        self.name = "IncSIR"
        self.attributes = [
            "infection_prob",
            "recovery_prob",
            "infection_type",
            "density",
            "init_param",
        ]


if __name__ == "__main__":
    unittest.main()
