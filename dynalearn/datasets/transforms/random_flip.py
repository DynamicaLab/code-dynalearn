import numpy as np

from dynalearn.datasets.transforms import StateTransform


class RandomFlipStateTransform(StateTransform):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.flip = config.flip
        Transform.__init__(self, config, **kwargs)

    def setup(self, experiment):
        self.num_states = experiment.dynamics.num_states

    def _transform_state_(self, x):
        _x = x.copy()
        num_nodes = x.shape[0]
        n = np.random.binomial(x.shape[0], self.flip)
        index = np.random.choice(range(num_nodes), size=n, replace=False)
        _x[index] = np.random.randint(self.num_states, size=n)
        return _x
