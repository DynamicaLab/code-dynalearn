import unittest
from .templates import *


class AttentionMetricsTest(MetricsTest, unittest.TestCase):
    @property
    def name(self):
        return "AttentionMetrics"

    def additional_configs(self):
        self.config.model.heads = 1


class AttentionStatesNMIMetricsTest(AttentionMetricsTest):
    @property
    def name(self):
        return "AttentionStatesNMIMetrics"


class AttentionNodeAttrNMIMetricsTest(AttentionMetricsTest):
    @property
    def name(self):
        return "AttentionNodeAttrNMIMetrics"


class AttentionEdgeAttrNMIMetricsTest(AttentionMetricsTest):
    @property
    def name(self):
        return "AttentionEdgeAttrNMIMetrics"
