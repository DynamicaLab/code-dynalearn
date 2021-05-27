import unittest
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from dynalearn.networks import GNPNetworkGenerator
from dynalearn.config import NetworkConfig


class ERNetworkTest(unittest.TestCase):
    def setUp(self):
        self.p = 0.5
        self.n = 100
        config = NetworkConfig.gnp(self.n, self.p)
        self.network = GNPNetworkGenerator(config)

    def test_generate(self):
        self.network.generate()


if __name__ == "__main__":
    unittest.main()
