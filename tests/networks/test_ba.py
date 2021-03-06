import unittest
import networkx as nx
import numpy as np

from dynalearn.networks import BANetworkGenerator
from dynalearn.config import NetworkConfig


class BANetworkTest(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.m = 2
        config = NetworkConfig.ba(self.n, self.m)
        self.network = BANetworkGenerator(config)

    def test_generate(self):
        self.network.generate()


if __name__ == "__main__":
    unittest.main()
