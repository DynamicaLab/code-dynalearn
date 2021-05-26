import numpy as np

from dynalearn.datasets.transforms import StateTransform


class RemapStateTransform(StateTransform):
    def setup(self, experiment):
        self.state_map = experiment.dynamics.state_map

    def _transform_state_(self, x):
        _x = np.vectorize(self.state_map.get)(x.copy())
        return _x


class PartiallyRemapStateTransform(StateTransform):
    def setup(self, experiment):
        self.state_map = experiment.dynamics.state_map
        self.hide_prob = experiment.dynamics.hide_prob

    def _transform_state_(self, x):
        if x.ndim == 1:
            window_size = 0
            num_nodes = x.shape[0]
        elif x.ndim == 2:
            window_size = x.shape[0]
            num_nodes = x.shape[1]
        _x = np.vectorize(self.state_map.get)(x.copy())
        y = x.copy()
        if window_size > 0:
            for i in range(window_size):
                n_remap = np.random.binomial(num_nodes, self.hide_prob)
                index = np.random.choice(range(num_nodes), size=n_remap, replace=False)
                y[i, index] = _x[i, index]
        else:
            n_remap = np.random.binomial(num_nodes, self.hide_prob)
            index = np.random.choice(range(num_nodes), size=n_remap, replace=False)
            y[index] = _x[index]
        return y
