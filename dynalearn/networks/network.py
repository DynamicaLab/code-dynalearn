import networkx as nx
import numpy as np

from dynalearn.util import (
    get_node_attr,
    get_edge_attr,
    set_node_attr,
    set_edge_attr,
)


class Network:
    def __init__(self, data=nx.Graph()):
        assert isinstance(data, nx.Graph)
        self.data = data

    def copy(self):
        return Network(data=self.data)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def node_attr(self):
        return self._node_attr

    @node_attr.setter
    def node_attr(self, node_attr):
        assert isinstance(node_attr, dict)
        for k, v in node_attr.items():
            assert isinstance(v, np.ndarray)
            assert len(v) == len(self._nodes)
            self._node_attr[k] = v
            v_dict = {n: v[n] for n in self.nodes}
            nx.set_node_attributes(self._data, v_dict, name=k)

    @property
    def edge_attr(self):
        return self._edge_attr

    @edge_attr.setter
    def edge_attr(self, edge_attr):
        assert isinstance(edge_attr, dict)
        for k, v in edge_attr.items():
            assert isinstance(v, np.ndarray)
            assert len(v) == len(self._edges)
            self._edge_attr[k] = v
            v_dict = {(n, m): v[i] for i, (n, m) in enumerate(self.edges)}
            nx.set_edge_attributes(self._data, v_dict, name=k)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._nodes = self._get_nodes_(data)
        self._edges = self._get_edges_(data)
        self._node_attr = self._get_node_attr_(data)
        self._edge_attr = self._get_edge_attr_(data)

    def to_directed(self):
        data = self.data.to_directed()
        return Network(data=data)

    def to_array(self):
        return nx.to_numpy_array(self.data)

    def get_node_data(self):
        n = self.number_of_nodes()
        node_data = np.zeros((n, 0))
        for k, v in self.node_attr.items():
            node_data = np.concatenate([node_data, v.reshape(n, -1)], axis=-1)
        return node_data

    def get_edge_data(self):
        m = self.number_of_edges()
        edge_data = np.zeros((m, 0))
        for k, v in self.edge_attr.items():
            edge_data = np.concatenate([edge_data, v.reshape(m, -1)], axis=-1)
        if edge_data is None:
            return np.zeros((m, 0))
        else:
            return edge_data

    def degree(self, index=None):
        degree = np.array(list(dict(self.data.degree()).values()))
        if index is None:
            return degree
        else:
            return degree[index]

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return self.data.number_of_edges()

    def neighbors(self, index):
        return np.array(list(self.data.neighbors(index)))

    def _get_nodes_(self, g):
        return np.array(list(g.nodes()))

    def _get_edges_(self, g):
        _g = g.to_directed()
        return np.array(list(_g.edges()))

    def _get_node_attr_(self, g):
        return get_node_attr(g)

    def _get_edge_attr_(self, g):
        return get_edge_attr(g)


class MultiplexNetwork:
    def __init__(self, data={}):
        assert isinstance(data, dict)
        self.data = data

    def copy(self):
        return MultiplexNetwork(data=self.data)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def node_attr(self):
        return self._node_attr

    @node_attr.setter
    def node_attr(self, node_attr):
        assert isinstance(node_attr, dict)
        for k, v in node_attr.items():
            assert isinstance(v, np.ndarray)
            assert len(v) == len(self._nodes)
            self._node_attr[k] = v
            v_dict = {n: v[n] for n in self.nodes}
            for l in self.layers:
                nx.set_node_attributes(self._data[l], v_dict, name=k)

    @property
    def edge_attr(self):
        return self._edge_attr

    @edge_attr.setter
    def edge_attr(self, edge_attr):
        assert isinstance(edge_attr, dict)
        for l in edge_attr.keys():
            assert l in self.data
            for k, v in edge_attr[l].items():
                assert isinstance(v, np.ndarray)
                assert len(v) == len(self._edges)
                self._edge_attr[l][k] = v
                v_dict = {(n, m): v[i] for i, (n, m) in enumerate(self.edges[l])}
                nx.set_edge_attributes(self.data[l], v, name=k)

    @property
    def layers(self):
        return list(self.edges.keys())

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert isinstance(data, dict)
        self._data = data
        self._nodes = self._get_nodes_(data)
        self._edges = self._get_edges_(data)
        self._node_attr = self._get_node_attr_(data)
        self._edge_attr = self._get_edge_attr_(data)

    def to_directed(self):
        return MultiplexNetwork({k: v.to_directed() for k, v in self.data.items()})

    def collapse(self, layer=None):
        if layer is None:
            layer = []
        if not isinstance(layer, list):
            layer = [layer]
        elif len(layer) == 0:
            layer = self.layers

        g = nx.empty_graph(create_using=nx.DiGraph())
        g.add_nodes_from(self.nodes)
        g = set_node_attr(g, self.node_attr)
        for k in layer:
            edge_list = self.edges[k]
            edge_attr = self.edge_attr[k]
            g.add_edges_from(edge_list)

            for kk, vv in edge_attr.items():
                for i, (u, v) in enumerate(edge_list):
                    if kk in g.edges[u, v]:
                        g.edges[u, v][kk] += vv[i]
                    else:
                        g.edges[u, v][kk] = vv[i]
        return Network(data=g)

    def to_array(self):
        return {l: nx.to_numpy_array(self.data[l]) for l in self.layers}

    def get_node_data(self):
        n = self.number_of_nodes()
        node_data = np.zeros((n, 0))
        for k, v in self.node_attr.items():
            node_data = np.concatenate([node_data, v.reshape(n, -1)], axis=-1)
        return node_data

    def get_edge_data(self):
        edge_data = {}
        for k, v in self.edge_attr.items():
            m = len(self.edges[k])
            edge_data[k] = np.zeros((m, 0))
            for kk, vv in v.items():
                edge_data[k] = np.concatenate(
                    [edge_data[k], vv.reshape(m, -1)], axis=-1
                )
        return edge_data

    def degree(self):
        return {
            k: np.array(list(dict(v.degree()).values())) for k, v in self.data.items()
        }

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return {k: v.number_of_edges() for k, v in self.data.items()}

    def neighbors(self, index):
        return {k: np.array(list(v.neighbors(index))) for k, v in self.data.items()}

    def _get_nodes_(self, g):
        assert isinstance(g, dict)
        nodes = set()
        for k, v in g.items():
            nodes = nodes.union(set(v.nodes()))
        return np.array(list(nodes))

    def _get_edges_(self, g):
        assert isinstance(g, dict)
        edges = {}
        for k, v in g.items():
            edges[k] = np.array(list(v.edges()))
        return edges

    def _get_node_attr_(self, g):
        assert isinstance(g, dict)
        all_node_attr = get_node_attr(g)
        node_attr = None
        for l, na in all_node_attr.items():
            if node_attr is None:
                node_attr = na
            for k, v in na.items():
                if k in node_attr:
                    assert isinstance(v, np.ndarray)
                    assert isinstance(na[k], np.ndarray)
                    assert np.all(na[k] == v)
                else:
                    node_attr[k] = v
        return node_attr

    def _get_edge_attr_(self, g):
        assert isinstance(g, dict)
        return get_edge_attr(g)
