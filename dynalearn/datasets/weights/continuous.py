import networkx as nx
import numpy as np

from scipy.stats import gmean
from .weight import Weight
from .kde import KernelDensityEstimator


class ContinuousStateWeight(Weight):
    def __init__(self, name="weights", reduce=True, bias=1.0):
        self.reduce = reduce
        Weight.__init__(self, name=name, max_num_samples=10000, bias=bias)

    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [dataset.inputs[i].data.shape[0] for i in range(dataset.networks.size)]
        )

    def _reduce_node_state_(self, index, states, network):
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        return x

    def _reduce_total_state_(self, states, network):
        return

    def _reduce_node_(self, index, network):
        return

    def _reduce_network_(self, network):
        return

    def _get_features_(self, network, states, pb=None):
        x = self._reduce_network_(network)
        if x is not None:
            self._add_features_("network", x)
        for i in network.nodes:
            k = network.degree(i)
            self._add_features_(("degree", int(k)))
            x = self._reduce_node_(i, network)
            if x is not None:
                self._add_features_(("node", int(k)), x)

        for i, s in enumerate(states):
            y = self._reduce_total_state_(s, network)
            if y is not None:
                self._add_features_("total_state", y)
            for j, ss in enumerate(s):
                k = network.degree(j)
                x = self._reduce_node_state_(j, s, network)
                if x is not None:
                    self._add_features_(("node_state", int(k)), x)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))
        z = 0
        kde = {}
        pp = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            else:
                kde[k] = KernelDensityEstimator(
                    samples=v, max_num_samples=self.max_num_samples
                )
        g_feats = self._reduce_network_(network)
        if g_feats is not None:
            p_g = kde["network"].pdf(g_feats)
        else:
            p_g = 1.0
        for i, s in enumerate(states):
            s_feats = self._reduce_total_state_(s, network)
            s_feats = self._reduce_total_state_(s, network)
            if s_feats is not None:
                p_s = kde["total_state"].pdf(s_feats)
            else:
                p_s = 1.0
            for j, ss in enumerate(s):
                k = network.degree(j)
                p_k = self.features[("degree", k)] / z

                ss_feats = self._reduce_node_state_(j, s, network)
                if ss_feats is not None:
                    p_ss = gmean(kde[("node_state", k)].pdf(ss_feats))
                else:
                    p_ss = 1.0

                n_feats = self._reduce_node_(j, network)
                if n_feats is not None:
                    p_n = kde[("node", k)].pdf(n_feats)
                else:
                    p_n = 1.0

                weights[i, j] = p_k * p_s * p_ss * p_n * p_g
            if pb is not None:
                pb.update()
        return weights


class ContinuousGlobalStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        return

    def _reduce_total_state_(self, states, network):
        return states.sum(0).reshape(-1)


class StrengthContinuousGlobalStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        return

    def _reduce_total_state_(self, states, network):
        return states.sum(0).reshape(-1)

    def _reduce_node_(self, index, network):
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.data.edges[index, l]:
                s += network.data.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        return s.reshape(-1)


class StrengthContinuousStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = states[index].reshape(-1)
        if self.reduce:
            x = np.array([x.sum()])
        s = np.array([0.0])
        for l in network.neighbors(index):
            if "weight" in network.data.edges[index, l]:
                s += network.data.edges[index, l]["weight"]
            else:
                s += np.array([1.0])
        return np.concatenate([x, s])


class ContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = []
        _x = states[index].reshape(-1)
        if self.reduce:
            _x = np.array([_x.sum()])
        for l in network.neighbors(index):
            _y = states[l].reshape(-1)
            if self.reduce:
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y]))
        return x


class StrengthContinuousCompoundStateWeight(ContinuousStateWeight):
    def _reduce_node_state_(self, index, states, network):
        x = []
        s = states[index]
        for l in network.neighbors(index):
            _x = s.reshape(-1)
            _y = states[l].reshape(-1)
            if "weight" in network.data.edges[index, l]:
                _w = np.array([network.data.edges[index, l]["weight"]])
            else:
                _w = np.array([1.0])
            if self.reduce:
                _x = np.array([_x.sum()])
                _y = np.array([_y.sum()])
            x.append(np.concatenate([_x, _y, _w]))
        return x
