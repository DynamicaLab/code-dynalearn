"""

dynamic.py

Created by Charles Murphy on 26-06-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynamicalNetwork which generate network on which a dynamical
process occurs.

"""

import networkx as nx
import numpy as np
import pickle
import torch as pt
import os

from math import ceil
from dynalearn.networks import Network, MultiplexNetwork
from dynalearn.util import (
    to_edge_index,
    get_edge_attr,
    get_node_strength,
    collapse_networks,
)


class Dynamics:
    def __init__(self, config, num_states):
        self._config = config
        self._num_states = num_states
        self._network = None
        self._edge_index = None
        self._num_nodes = None

        self.lag = self._config.lag = config.__dict__.pop("lag", 1)
        self.lagstep = self._config.lagstep = config.__dict__.pop("lagstep", 1)

    def initial_state(self):
        raise NotImplementedError("self.initial_state() has not been impletemented")

    def predict(self, x):
        raise NotImplementedError("self.predict() has not been impletemented")

    def loglikelihood(self, x):
        raise NotImplementedError("self.loglikelihood() has not been impletemented")

    def sample(self, x):
        raise NotImplementedError("sample has not been impletemented")

    def is_dead(self, x):
        raise NotImplementedError("is_dead has not been impletemented")

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        assert isinstance(network, Network)
        network = network.to_directed()
        self._edge_index = network.edges.T
        self._node_degree = network.degree()
        self._num_nodes = network.number_of_nodes()
        self._network = network
        self.update_edge_attr()
        self.update_node_attr()

    @property
    def edge_index(self):
        if self._edge_index is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_index

    @property
    def node_degree(self):
        if self._node_degree is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_degree

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._num_nodes

    @property
    def num_states(self):
        return self._num_states

    def update_node_attr(self):
        return

    def update_edge_attr(self):
        return


class WeightedDynamics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        self._edge_weight = None
        self._node_strength = None

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        assert isinstance(network, Network)
        network = network.to_directed()
        self._edge_index = network.edges.T
        self._edge_weight = network.edge_attr["weight"].reshape(-1, 1)
        self._node_degree = network.degree()
        self._num_nodes = network.number_of_nodes()
        self._node_strength = np.zeros(self._num_nodes)
        for i, (u, v) in enumerate(network.edges):
            self._node_strength[u] += self._edge_weight[i]
        self._network = network
        self.update_edge_attr()
        self.update_node_attr()

    @property
    def edge_weight(self):
        if self._edge_weight is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_weight

    @property
    def node_strength(self):
        if self._node_strength is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_strength


class MultiplexDynamics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        self._collapsed_edge_index = None
        self._collapsed_node_degree = None

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        assert isinstance(network, MultiplexNetwork)
        network = network.to_directed()
        self._edge_index = {k: v.T for k, v in network.edges.items()}
        self._node_degree = network.degree()
        self._num_nodes = network.number_of_nodes()
        self._network = network
        self.update_edge_attr()
        self.update_node_attr()
        self.collapsed_network = network.collapse()

    @property
    def collapsed_network(self):
        if self._collapsed_network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_network

    @collapsed_network.setter
    def collapsed_network(self, network):
        assert isinstance(network, Network)
        network = network.to_directed()
        self._collapsed_network = network
        self._collapsed_edge_index = network.edges.T
        self._collapsed_node_degree = network.degree()

    @property
    def collapsed_edge_index(self):
        if self._collapsed_edge_index is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_edge_index

    @property
    def collapsed_node_degree(self):
        if self._collapsed_node_degree is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_node_degree


class WeightedMultiplexDynamics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        self._edge_weight = None
        self._node_strength = None
        self._collapsed_network = None
        self._collapsed_edge_index = None
        self._collapsed_node_degree = None
        self._collapsed_node_strength = None
        self._collapsed_edge_weight = None

    @property
    def network(self):
        if self._network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._network

    @network.setter
    def network(self, network):
        assert isinstance(network, MultiplexNetwork)
        network = network.to_directed()
        self._edge_index = {k: v.T for k, v in network.edges.items()}
        self._node_degree = network.degree()
        self._num_nodes = network.number_of_nodes()
        self._edge_weight = {
            k: v["weight"].reshape(-1, 1) for k, v in network.edge_attr.items()
        }
        self._node_strength = {}
        for k, edges in network.edges.items():
            self._node_strength[k] = np.zeros(self._num_nodes)
            for i, (u, v) in enumerate(edges):
                self._node_strength[k][u] += self._edge_weight[k][i]
        self._network = network
        self.update_edge_attr()
        self.update_node_attr()
        self.collapsed_network = network.collapse()

    @property
    def collapsed_network(self):
        if self._collapsed_network is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_network

    @collapsed_network.setter
    def collapsed_network(self, network):
        assert isinstance(network, Network)
        assert network.number_of_nodes() == self._num_nodes
        network = network.to_directed()
        self._collapsed_network = network
        self._collapsed_edge_index = network.edges.T
        self._collapsed_edge_weight = network.edge_attr["weight"].reshape(-1, 1)
        self._collapsed_node_degree = network.degree()
        self._collapsed_node_strength = np.zeros(self._num_nodes)
        for i, (u, v) in enumerate(network.edges):
            self._collapsed_node_strength[u] += self._collapsed_edge_weight[i]

    @property
    def edge_weight(self):
        if self._edge_weight is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._edge_weight

    @property
    def node_strength(self):
        if self._node_strength is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._node_strength

    @property
    def collapsed_edge_index(self):
        if self._collapsed_edge_index is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_edge_index

    @property
    def collapsed_node_degree(self):
        if self._collapsed_node_degree is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_node_degree

    @property
    def collapsed_edge_weight(self):
        if self._collapsed_edge_weight is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_edge_weight

    @property
    def collapsed_node_strength(self):
        if self._collapsed_node_strength is None:
            raise ValueError("No network has been parsed to the dynamics.")
        else:
            return self._collapsed_node_strength
