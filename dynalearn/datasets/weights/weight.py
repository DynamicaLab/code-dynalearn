import networkx as nx
import numpy as np
import tqdm

from dynalearn.datasets.data import DataCollection, StateData
from dynalearn.networks import Network, MultiplexNetwork
from dynalearn.util import Verbose


class Weight(DataCollection):
    def __init__(self, name="weights", max_num_samples=-1, bias=1.0):
        DataCollection.__init__(self, name=name, template=StateData)
        self.max_num_samples = max_num_samples
        self.bias = bias
        self.features = {}

    def check_state(self, state):
        if not isinstance(state, np.ndarray):
            raise TypeError(
                f"Invalid type {type(state)} for `state`, expected `np.ndarray`."
            )

    def check_network(self, network):
        if not isinstance(network, (Network, MultiplexNetwork)):
            raise TypeError(
                f"Invalid type {type(network)} for `network`, expected [`Network`, `MultiplexNetwork`]."
            )

    def _get_features_(self, network, states, pb=None):
        if pb is not None:
            pb.update()
        return

    def _get_weights_(self, network, states, pb=None):
        if pb is not None:
            pb.update()
        return np.ones((states.shape[0], states.shape[1]))

    def compute(self, dataset, verbose=Verbose()):
        self.setUp(dataset)
        pb = verbose.progress_bar("Computing weights", self.num_updates)
        self.compute_features(dataset, pb=pb)
        self.compute_weights(dataset, pb=pb)
        self.clear()
        if pb is not None:
            pb.close()

    def setUp(self, dataset):
        self.num_updates = 2 * dataset.networks.size

    def compute_features(self, dataset, pb=None):
        for i in range(dataset.networks.size):
            x = dataset.inputs[i].data
            g = dataset.networks[i].data
            if isinstance(g, MultiplexNetwork):
                g = g.collapse()
            self.check_network(g)
            self.check_state(x)
            self._get_features_(g, x, pb=pb)
        return

    def compute_weights(self, dataset, pb=None):
        weights = []
        for i in range(dataset.networks.size):
            x = dataset.inputs[i].data
            g = dataset.networks[i].data
            if isinstance(g, MultiplexNetwork):
                g = g.collapse()
            self.check_network(g)
            self.check_state(x)
            w = self._get_weights_(g, x, pb=pb) ** (-self.bias)
            weights = StateData(data=w)
            self.add(weights)

    def _add_features_(self, key, value=None):
        if value is None:
            if key not in self.features:
                self.features[key] = 1
            else:
                self.features[key] += 1
        else:
            if key not in self.features:
                if isinstance(value, list):
                    self.features[key] = value
                else:
                    self.features[key] = [value]
            else:
                if isinstance(value, list):
                    self.features[key].extend(value)
                else:
                    self.features[key].append(value)

    def clear(self):
        self.features = {}

    def to_state_weights(self):
        state_weights = DataCollection()
        for i in range(self.size):
            w = self.data_list[i].data.sum(-1)
            state_weights.add(StateData(data=w))
        return state_weights

    def to_network_weights(self):
        network_weights = StateData()
        weight = []
        for i in range(self.size):
            w = self.data_list[i].data.sum()
            weight.append(w)
        network_weights.data = np.array(weight)
        return network_weights
