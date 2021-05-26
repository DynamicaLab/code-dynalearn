import networkx as nx
import numpy as np


class Sampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.config = dataset.config
        self.bias = self.config.bias
        self.replace = self.config.replace
        self.counter = 0
        self.avail_networks = list()
        self.avail_states = dict()

    def __call__(self):
        if len(self.avail_networks) > 0 and self.counter <= len(self.dataset):
            g_index = self._get_network_()
            s_index = self._get_state_(g_index)
            self.update(g_index, s_index)
            return (g_index, s_index)
        else:
            self.reset()
            raise StopIteration

    def update(self, g_index, s_index):
        self.counter += 1
        if not self.replace:
            self.avail_states[g_index].remove(s_index)
        if len(self.avail_states[g_index]) == 0:
            self.avail_networks.remove(g_index)

    def reset(self):
        self.counter = 0
        self.avail_networks = list(range(self.dataset.network_weights.size))
        self.avail_states = {
            i: list(range(int(self.dataset.state_weights[i].data.shape[0])))
            for i in self.avail_networks
        }

    def _get_network_(self):
        indices = self.avail_networks
        p = self.dataset.network_weights.data[indices]
        p /= p.sum()
        index = np.random.choice(self.avail_networks, p=p)
        return index

    def _get_state_(self, g_index):
        indices = self.avail_states[g_index]
        p = self.dataset.state_weights[g_index].data[indices]
        p /= p.sum()
        index = np.random.choice(indices, p=p)
        return index
