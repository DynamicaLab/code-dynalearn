import networkx as nx
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from dynalearn.config import NetworkConfig
from dynalearn.datasets import ThresholdNetworkTransform, NetworkData
from dynalearn.networks.getter import get as get_network
from dynalearn.networks import Network, MultiplexNetwork
from unittest import TestCase


class ThresholdNetworkTransformTest(TestCase):
    def setUp(self):
        self.threshold = 10
        self.collapse = True

        self.network = get_network(NetworkConfig.w_ba())
        self.multiplex_network = get_network(NetworkConfig.mw_ba(layers=3))

        self.transform_coll = ThresholdNetworkTransform(
            threshold=self.threshold, collapse=True
        )
        self.transform_ncoll = ThresholdNetworkTransform(
            threshold=self.threshold, collapse=False
        )
        return

    def test_call(self):
        g = self.network.generate()
        x = NetworkData(data=g)
        x = self.transform_coll(x)
        self.assertTrue(isinstance(x.data, Network))

        g = self.multiplex_network.generate()
        x = NetworkData(data=g)
        x = self.transform_coll(x)
        self.assertTrue(isinstance(x.data, Network))

        g = self.multiplex_network.generate()
        x = NetworkData(data=g)
        x = self.transform_ncoll(x)
        self.assertTrue(isinstance(x.data, MultiplexNetwork))


if __name__ == "__main__":
    unittest.main()
