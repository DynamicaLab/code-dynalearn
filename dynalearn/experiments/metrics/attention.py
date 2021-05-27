import networkx as nx
import numpy as np

from abc import abstractmethod
from functools import partial
from sklearn.feature_selection import mutual_info_regression
from dynalearn.experiments.metrics import Metrics
from dynalearn.util import Verbose
from dynalearn.nn.models import DynamicsGATConv, Kapoor2020GNN
from dynalearn.networks import MultiplexNetwork
from .util.mutual_info import mutual_info


class AttentionMetrics(Metrics):
    def __init__(self, config):
        Metrics.__init__(self, config)
        self.max_num_points = config.attention.get("max_num_points", 100)
        self.indices = {}

    def initialize(self, experiment):
        self.model = experiment.model
        self.dataset = experiment.dataset
        self.indices = self._get_indices_()
        self.num_updates = len(self.indices)
        if self.model.config.is_multiplex:
            layers = self.model.config.network_layers
            for l in layers:
                gnn = getattr(self.model.nn.gnn_layer, f"layer_{l}")
                if not isinstance(gnn, DynamicsGATConv):
                    return
        else:
            if isinstance(self.model.nn, Kapoor2020GNN) or not isinstance(
                self.model.nn.gnn_layer, DynamicsGATConv
            ):
                return
            layers = [None]
        for l in layers:
            suffix = ""
            if l is not None:
                suffix += f"-{l}"
            self.names.append("attcoeffs" + suffix)
            self.get_data["attcoeffs" + suffix] = partial(self.get_attcoeffs, layer=l)

            self.names.append("states" + suffix)
            self.get_data["states" + suffix] = partial(self.get_states, layer=l)

            self.names.append("nodeattr" + suffix)
            self.get_data["nodeattr" + suffix] = partial(self.get_nodeattr, layer=l)

            self.names.append("edgeattr" + suffix)
            self.get_data["edgeattr" + suffix] = partial(self.get_edgeattr, layer=l)

    def _get_indices_(self, doall=True):
        inputs = self.dataset.inputs[0].data
        T = inputs.shape[0]
        all_indices = np.arange(T)
        if not doall:
            weights = self.dataset.state_weights[0].data
            all_indices = all_indices[weights > 0]
        num_points = min(T, self.max_num_points)
        return np.random.choice(all_indices, size=num_points, replace=False)

    def get_attcoeffs(self, layer=None, pb=None):
        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        gnn = self.model.nn.gnn_layer
        edge_index, edge_attr, node_attr = self.model.nn.transformers[
            "t_networks"
        ].forward(network)

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]
            edge_layers = getattr(self.model.nn.edge_layers, f"layer_{layer}")
            gnn = getattr(gnn, f"layer_{layer}")
            M = network.edges[layer].shape[0]
        else:
            edge_layers = self.model.nn.edge_layers
            M = network.edges.shape[0]
        T = inputs.shape[0]

        if node_attr is not None:
            node_attr = self.model.nn.node_layers(node_attr) * 1
        if edge_attr is not None:
            edge_attr = edge_layers(edge_attr) * 1

        results = np.zeros((inputs.shape[0], edge_index.shape[1], gnn.heads))
        for i, x in enumerate(inputs):
            x = self.model.nn.transformers["t_inputs"].forward(x)
            x = self.model.nn.in_layers(x)
            x = self.model.nn.merge_nodeattr(x, node_attr)
            out = gnn.forward(x, edge_index, edge_attr, return_attention_weights=True)
            results[i] = out[1][1].detach().cpu().numpy()
            if pb is not None:
                pb.update()
        return results.reshape(T * M, -1)

    def get_states(self, layer=None, pb=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]
        T = inputs.shape[0]
        M = edge_index.shape[0]
        if inputs.ndim == 4:
            D = inputs.shape[-2]
        else:
            D = 1
        results = np.zeros((T, M, 2, D))
        for i, x in enumerate(inputs):
            t, s = edge_index.T
            x = x.T[-1].T
            sources, targets = np.expand_dims(x[s], 1), np.expand_dims(x[t], 1)
            if sources.ndim == 2:
                sources = np.expand_dims(sources, -1)
                targets = np.expand_dims(targets, -1)
            results[i] = np.concatenate((sources, targets), axis=1)
        results = results.reshape(T * M, 2, -1)
        return {
            "all": results.reshape(T * M, -1),
            "source": results[:, 0, :],
            "target": results[:, 1, :],
        }

    def get_nodeattr(self, layer=None, pb=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        s, t = edge_index.T
        res = {}
        for k, v in node_attr.items():
            sources, targets = np.expand_dims(v[s], 1), np.expand_dims(v[t], 1)
            r = np.concatenate((sources, targets), axis=1)
            res[k] = r.reshape(1, *r.shape).repeat(T, axis=0).reshape(T * M, 2, -1)
        results = {"all-" + k: v.reshape(T * M, -1) for k, v in res.items()}
        results.update({"source-" + k: v[:, 0, :] for k, v in res.items()})
        results.update({"target-" + k: v[:, 1, :] for k, v in res.items()})
        return results

    def get_edgeattr(self, layer=None, pb=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        s, t = edge_index.T
        results = {}
        for k, v in edge_attr.items():
            results[k] = v.reshape(1, *v.shape).repeat(T, axis=0).reshape(T * M, -1)
        return results


class AttentionFeatureNMIMetrics(AttentionMetrics):
    def __init__(self, config):
        AttentionMetrics.__init__(self, config)
        p = config.__dict__.copy()
        self.n_neighbors = p.get("n_neighbors", 3)
        self.metric = p.get("metric", "euclidean")
        self.fname = None

    def initialize(self, experiment):
        self.model = experiment.model
        self.dataset = experiment.dataset
        self.indices = self._get_indices_()
        if self.model.config.is_multiplex:
            layers = self.model.config.network_layers
            for l in layers:
                gnn = getattr(self.model.nn.gnn_layer, f"layer_{l}")
                if not isinstance(gnn, DynamicsGATConv):
                    return
        else:
            if isinstance(self.model.nn, Kapoor2020GNN) or not isinstance(
                self.model.nn.gnn_layer, DynamicsGATConv
            ):
                return
            layers = [None]
        for l in layers:
            name = "nmi-"
            if l is not None:
                name = f"{l}-" + name
            attcoeffs = self.get_attcoeffs(layer=l)
            features = self.get_feature(layer=l)
            if isinstance(features, dict):
                for k, v in features.items():
                    d_name = name + f"att_vs_{self.fname}-{k}"
                    self.names.append(d_name)
                    self.get_data[d_name] = partial(self.compute_nmi, v, attcoeffs)
            else:
                d_name = name + f"att_vs_{self.fname}"
                self.names.append(d_name)
                self.get_data[d_name] = partial(self.compute_nmi, features, attcoeffs)

    def get_feature(self, layer=None):
        raise NotImplemented

    def compute_nmi(self, x, y, pb=None):
        mi = mutual_info(x, y, n_neighbors=self.n_neighbors, metric=self.metric)
        hx = mutual_info(x, x, n_neighbors=self.n_neighbors, metric=self.metric)
        hy = mutual_info(y, y, n_neighbors=self.n_neighbors, metric=self.metric)
        if hx == 0 and hy == 0:
            return 0.0
        return 2 * mi / (hx + hy)


class AttentionStatesNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "states"
        self.get_feature = self.get_states


class AttentionNodeAttrNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "nodeattr"
        self.get_feature = self.get_nodeattr


class AttentionEdgeAttrNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "edgeattr"
        self.get_feature = self.get_edgeattr
