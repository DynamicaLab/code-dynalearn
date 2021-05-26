import unittest
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")

from dynalearn.config.util import TrainingConfig
from templates import TemplateConfigTest

TRAIN_ATTRIBUTES = [
    "val_fraction",
    "val_bias",
    "epochs",
    "batch_size",
    "num_networks",
    "num_samples",
    "resampling",
    "maxlag",
    "resample_when_dead",
]


class DiscreteTrainingConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig.discrete()
        self.attributes = TRAIN_ATTRIBUTES


class ContinuousTrainingConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig.continuous()
        self.attributes = TRAIN_ATTRIBUTES
