import unittest
import networkx as nx
import numpy as np
import time

from templates import *
from dynalearn.dynamics import SIS, SIR, SISSIS, PlanckSIS
from dynalearn.config import DynamicsConfig, NetworkConfig
from dynalearn.networks.getter import get as get_network


class SISTest(StoContTemplateTest, unittest.TestCase):
    def get_model(self):
        self._num_states = 2
        return SIS(DynamicsConfig.sis())


class SIRTest(StoContTemplateTest, unittest.TestCase):
    def get_model(self):
        self._num_states = 3
        return SIR(DynamicsConfig.sir())


class SISSISTest(StoContTemplateTest, unittest.TestCase):
    def get_model(self):
        self._num_states = 4
        return SISSIS(DynamicsConfig.sissis())


class PlanckSISTest(StoContTemplateTest, unittest.TestCase):
    def get_model(self):
        self._num_states = 2
        return PlanckSIS(DynamicsConfig.plancksis())


if __name__ == "__main__":
    unittest.main()
