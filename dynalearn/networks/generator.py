import numpy as np
import networkx as nx

from dynalearn.config import Config


class NetworkGenerator:
    def __init__(self, config=None):
        config = config or Config()
        self._config = config
        self.is_weighted = False
        self.is_multiplex = False
        if "num_nodes" in config.__dict__:
            self.num_nodes = config.num_nodes
        else:
            self.num_nodes = None
        if "layers" in config.__dict__:
            self.layers = config.layers
            self.is_multiplex = True
        else:
            self.layers = None

    def generate(self, seed):
        raise NotImplementedError()

    @property
    def config(self):
        return self._config
