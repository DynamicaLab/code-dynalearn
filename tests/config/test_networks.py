import unittest
import warnings

warnings.filterwarnings("ignore")

from dynalearn.config import NetworkConfig
from templates import TemplateConfigTest


class GNPConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig.gnp(num_nodes=1000, p=0.004)
        self.attributes = ["num_nodes", "p"]
        self.name = "GNPNetworkGenerator"


class WGNPConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig.w_gnp(num_nodes=1000, p=0.004)
        self.attributes = ["num_nodes", "p", "weights", "transforms"]
        self.name = "GNPNetworkGenerator"


class BAConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig.ba(num_nodes=1000, m=2)
        self.attributes = ["num_nodes", "m"]
        self.name = "BANetworkGenerator"


class WBAConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig.w_ba(num_nodes=1000, m=2)
        self.attributes = ["num_nodes", "m", "weights", "transforms"]
        self.name = "BANetworkGenerator"


class MWBAConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig.mw_ba(num_nodes=1000, m=2)
        self.attributes = ["num_nodes", "m", "weights", "transforms", "layers"]
        self.name = "BANetworkGenerator"
