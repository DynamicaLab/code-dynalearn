import h5py
import networkx as nx
import numpy as np
import torch

from dynalearn.datasets.data.data import Data
from dynalearn.networks import Network, MultiplexNetwork
from dynalearn.util import (
    to_edge_index,
    get_edge_attr,
    set_edge_attr,
    get_node_attr,
    set_node_attr,
    onehot,
)


class NetworkData(Data):
    def __init__(self, name="network_data", data=None):
        Data.__init__(self, name=name)
        if data is not None:
            if isinstance(data, h5py.Group):
                data = self._load_graph_(data)
                if isinstance(data, dict):
                    data = MultiplexNetwork(data)
                elif isinstance(data, nx.Graph):
                    data = Network(data)
            assert issubclass(data.__class__, (Network, MultiplexNetwork))
            self.data = data
        else:
            self._data = None

    def __eq__(self, other):
        if isinstance(other, NetworkData):
            if isinstance(self.data, MultiplexNetwork):
                for l in self.data.layers:
                    if l not in other.data.layers:
                        return False
                    elif np.any(self.data.edges[l] != other.data.edges[k]):
                        return False
                return True
            else:
                return np.all(self.data.edges == other.data.edges)
        else:
            return False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def get(self):
        return self.data

    def save(self, h5file):
        group = h5file.create_group(self.name)
        if isinstance(self.data, MultiplexNetwork):
            for k in self.data.layers:
                _group = group.create_group(k)
                self._save_graph_(self.data.collapse(k), _group)
        else:
            self._save_graph_(self.data, group)

    def load(self, h5file):
        if self.name in h5file:
            group = h5file[self.name]
        else:
            group = h5file
        if "edge_list" in group:
            g = self._load_graph_(group)
            self.data = Network(data=g)
        else:
            g = {}
            for k in group.keys():
                if "edge_list" not in group:
                    raise ValueError(f"No edge list found while loading {self.name}.")
                g[k] = self._load_graph_(group[k])
            sefl.data = MultiplexNetwork(data=g)

    def _save_graph_(self, g, h5file):
        node_list = g.nodes
        node_attr = g.node_attr
        h5file.create_dataset("node_list", data=node_list)
        node_group = h5file.create_group("node_attr")
        for k, v in node_attr.items():
            node_group.create_dataset(k, data=v)

        if g.number_of_edges() > 0:
            edge_list = g.edges
            edge_attr = g.edge_attr
        else:
            edge_list = np.zeros((0, 2)).astype("int")
            edge_attr = {}
        h5file.create_dataset("edge_list", data=edge_list)
        edge_group = h5file.create_group("edge_attr")
        for k, v in edge_attr.items():
            edge_group.create_dataset(k, data=v)

    def _load_graph_(self, h5file):
        def load_g(h5group):
            node_list = h5group["node_list"][...]
            edge_list = h5group["edge_list"][...]

            g = nx.DiGraph()
            g.add_nodes_from(node_list)
            g.add_edges_from(edge_list)
            edge_attr = {}
            if "edge_attr" in h5group:
                for k, v in h5group["edge_attr"].items():
                    edge_attr[k] = v
                g = set_edge_attr(g, edge_attr)

            node_attr = {}
            if "node_attr" in h5group:
                for k, v in h5group["node_attr"].items():
                    node_attr[k] = v
                g = set_node_attr(g, node_attr)
            return g

        if "edge_list" in h5file:
            return load_g(h5file)
        else:
            return {k: load_g(v) for k, v in h5file.items()}
