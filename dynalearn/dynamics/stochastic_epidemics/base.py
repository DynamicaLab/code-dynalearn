import networkx as nx
import numpy as np
import torch

from dynalearn.dynamics.dynamics import Dynamics
from dynalearn.nn.models import Propagator
from dynalearn.util import from_binary, onehot


class StochasticEpidemics(Dynamics):
    def __init__(self, config, num_states):
        Dynamics.__init__(self, config, num_states)
        if "init_param" in config.__dict__:
            self.init_param = config.init_param
        else:
            self.init_param = None
        self.propagator = Propagator(num_states)
        self.state_map = {i: i for i in range(num_states)}

    def sample(self, x):
        p = self.predict(x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        y = np.array(dist.sample())
        return y

    def loglikelihood(self, x, y=None, g=None):
        if g is not None:
            self.network = g
        if y is None:
            y = np.roll(x, -1, axis=0)[:-1]
            x = x[:-1]

        if x.shape == (self.lag, self.num_nodes) or x.shape == (self.num_nodes):
            x = x.reshape(1, self.lag, self.num_nodes)
            y = y.reshape(1, self.num_nodes)

        loglikelihood = 0
        for i in range(x.shape[0]):
            p = self.predict(x[i])
            onehot_y = onehot(y[i], num_class=self.num_states)
            p = (onehot_y * p).sum(-1)
            p[p <= 1e-15] = 1e-15
            logp = np.log(p)
            loglikelihood += logp.sum()
        return loglikelihood

    def neighbors_state(self, x):
        if len(x.shape) > 1:
            raise ValueError(
                f"Invalid shape, expected shape of size 1 and got {x.shape}"
            )

        l = self.propagator.forward(x, self.edge_index)
        l = l.cpu().numpy()
        return l

    def initial_state(self, init_param=None, squeeze=True):
        if init_param is None:
            init_param = self.init_param
        if init_param is None:
            init_param = np.random.rand(self.num_states)
            init_param /= init_param.sum()
        elif isinstance(init_param, list):
            init_param = np.array(init_param)

        assert isinstance(init_param, np.ndarray)
        assert init_param.shape == (self.num_states,)
        x = np.random.multinomial(1, init_param, size=self.num_nodes)
        x = np.where(x == 1.0)[1]
        x = x.reshape(*x.shape, 1).repeat(self.lag, -1)
        if squeeze:
            return x.squeeze()
        else:
            return x

    def is_dead(self, x):
        if x.ndim == 2:
            x = x[:, -1]
        if self.number_of_infected(x) == 0:
            return True
        else:
            return False

    def nearly_dead_state(self, num_infected=None):
        raise NotImplementedError()

    def number_of_infected(self, x):
        raise NotImplementedError()
