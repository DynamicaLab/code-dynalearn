import unittest

from dynalearn.config import TrainableConfig
from .templates import TemplateConfigTest

GNN_ATTRIBUTES = [
    "gnn_name",
    "type",
    "num_states",
    "lag",
    "lagstep",
    "optimizer",
    "in_activation",
    "gnn_activation",
    "out_activation",
    "in_channels",
    "gnn_channels",
    "out_channels",
    "heads",
    "concat",
    "bias",
    "self_attention",
]

WGNN_ATTRIBUTES = GNN_ATTRIBUTES + [
    "weighted",
    "node_activation",
    "edge_activation",
    "node_channels",
    "edge_channels",
    "edge_gnn_channels",
]


class TrainableSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.sis()
        self.name = "GNNSEDynamics"
        self.attributes = GNN_ATTRIBUTES


class TrainablePlanckSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.plancksis()
        self.name = "GNNSEDynamics"
        self.attributes = GNN_ATTRIBUTES


class TrainableSISSISConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.sissis()
        self.name = "GNNSEDynamics"
        self.attributes = GNN_ATTRIBUTES


class TrainableDSIRConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.dsir()
        self.name = "GNNDEDynamics"
        self.attributes = WGNN_ATTRIBUTES


class TrainableIncSIRConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.incsir()
        self.name = "GNNIncidenceDynamics"
        self.attributes = WGNN_ATTRIBUTES


class TrainableKapoorConfigTest(TemplateConfigTest, unittest.TestCase):
    def setUp(self):
        self.config = TrainableConfig.kapoor()
        self.name = "KapoorDynamics"
        self.attributes = ["lag", "lagstep", "num_states", "optimizer"]
