import unittest

from dynalearn.config import DiscreteDatasetConfig, ContinuousDatasetConfig
from .templates import TemplateConfigTest


class DiscretePlainDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DiscreteDatasetConfig.plain()
        self.attributes = ["modes", "bias", "replace", "use_groundtruth"]
        self.name = "DiscreteDataset"


class DiscreteStructureDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DiscreteDatasetConfig.structure()
        self.attributes = [
            "modes",
            "bias",
            "replace",
            "use_groundtruth",
            "use_strength",
        ]
        self.name = "DiscreteStructureWeightDataset"


class DiscreteStateDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = DiscreteDatasetConfig.state()
        self.attributes = [
            "modes",
            "bias",
            "replace",
            "use_groundtruth",
            "use_strength",
            "compounded",
        ]
        self.name = "DiscreteStateWeightDataset"


class ContinuousPlainDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ContinuousDatasetConfig.plain()
        self.attributes = ["modes", "bias", "replace", "use_groundtruth"]
        self.name = "ContinuousDataset"


class ContinuousStructureDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ContinuousDatasetConfig.structure()
        self.attributes = [
            "modes",
            "bias",
            "replace",
            "use_groundtruth",
            "use_strength",
        ]
        self.name = "ContinuousStructureWeightDataset"


class ContinuousStateDatasetConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = ContinuousDatasetConfig.state()
        self.attributes = [
            "modes",
            "bias",
            "replace",
            "use_groundtruth",
            "use_strength",
            "compounded",
            "total",
            "reduce",
            "max_num_points",
        ]
        self.name = "ContinuousStateWeightDataset"
