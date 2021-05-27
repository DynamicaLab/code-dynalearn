import networkx as nx
import numpy as np
import os
import torch
import unittest

from .templates import TemplateDatasetTest
from dynalearn.config import ExperimentConfig


class ContinuousDatasetTest(TemplateDatasetTest, unittest.TestCase):
    def get_config(self):
        return ExperimentConfig.test(config="continuous")

    def get_input_shape(self):
        return self.num_nodes, self.num_states, self.lag


if __name__ == "__main__":
    unittest.main()
