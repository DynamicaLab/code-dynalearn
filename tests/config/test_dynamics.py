import unittest

from dynalearn.config import DynamicsConfig
from .templates import TemplateConfigTest


class SISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.sis()
        self.name = "SIS"
        self.attributes = [
            "infection",
            "recovery",
        ]


class PlanckSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DynamicsConfig.plancksis()
        self.name = "PlanckSIS"
        self.attributes = [
            "temperature",
            "recovery",
        ]


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
        ]


if __name__ == "__main__":
    unittest.main()
