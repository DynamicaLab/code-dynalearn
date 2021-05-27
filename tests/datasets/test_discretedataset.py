import networkx as nx
import numpy as np
import torch
import unittest

from .templates import TemplateDatasetTest
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment


class DiscreteDatasetTest(TemplateDatasetTest, unittest.TestCase):
    def get_config(self):
        return ExperimentConfig.test(config="discrete")

    def get_input_shape(self):
        return self.num_nodes, self.lag


if __name__ == "__main__":
    unittest.main()
