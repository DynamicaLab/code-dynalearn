import numpy as np
import torch

from .base import StochasticEpidemics
from dynalearn.datasets.transforms import RemapStateTransform
from dynalearn.dynamics.activation import independent
from dynalearn.config import Config
from dynalearn.util import onehot


class SISSIS(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        num_states = 4

        self.infection1 = config.infection1
        self.infection2 = config.infection2
        self.recovery1 = config.recovery1
        self.recovery2 = config.recovery2
        self.coupling = config.coupling
        assert (
            self.infection1 >= 0 and self.infection1 <= 1
        ), "Invalid parameter, infection1 must be between [0, 1]."
        assert (
            self.infection2 >= 0 and self.infection2 <= 1
        ), "Invalid parameter, infection2 must be between [0, 1]."
        assert (
            self.recovery1 >= 0 and self.recovery1 <= 1
        ), "Invalid parameter, recovery1 must be between [0, 1]."
        assert (
            self.recovery2 >= 0 and self.recovery2 <= 1
        ), "Invalid parameter, recovery2 must be between [0, 1]."
        assert (
            self.coupling >= 0
            and self.coupling * self.infection1 < 1
            and self.coupling * self.infection2 < 1
        ), "Invalid parameter, coupling must be greater than 0 and bounded."

        StochasticEpidemics.__init__(self, config, num_states)

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        l = self.neighbors_state(x)
        p0, p1 = self.infection(x, l)
        q0, q1 = self.recovery(x, l)

        ltp = np.zeros((x.shape[0], self.num_states))

        # SS nodes
        ltp[x == 0, 0] = (1 - p0[x == 0]) * (1 - p1[x == 0])
        ltp[x == 0, 1] = p0[x == 0] * (1 - p1[x == 0])
        ltp[x == 0, 2] = (1 - p0[x == 0]) * p1[x == 0]
        ltp[x == 0, 3] = p0[x == 0] * p1[x == 0]

        # IS nodes
        ltp[x == 1, 0] = q0[x == 1] * (1 - p1[x == 1])
        ltp[x == 1, 1] = (1 - q0[x == 1]) * (1 - p1[x == 1])
        ltp[x == 1, 2] = q0[x == 1] * p1[x == 1]
        ltp[x == 1, 3] = (1 - q0[x == 1]) * p1[x == 1]

        # SI nodes
        ltp[x == 2, 0] = (1 - p0[x == 2]) * q1[x == 2]
        ltp[x == 2, 1] = p0[x == 2] * q1[x == 2]
        ltp[x == 2, 2] = (1 - p0[x == 2]) * (1 - q1[x == 2])
        ltp[x == 2, 3] = p0[x == 2] * (1 - q1[x == 2])

        # II nodes
        ltp[x == 3, 0] = q0[x == 3] * q1[x == 3]
        ltp[x == 3, 1] = (1 - q0[x == 3]) * q1[x == 3]
        ltp[x == 3, 2] = q0[x == 3] * (1 - q1[x == 3])
        ltp[x == 3, 3] = (1 - q0[x == 3]) * (1 - q1[x == 3])
        return ltp

    def infection(self, x, l):

        inf0 = np.zeros(x.shape)
        inf1 = np.zeros(x.shape)

        # Node SS
        inf0[x == 0] = (
            1
            - (1 - self.infection1) ** l[1, x == 0]
            * (1 - self.coupling * self.infection1) ** l[3, x == 0]
        )
        inf1[x == 0] = (
            1
            - (1 - self.infection2) ** l[2, x == 0]
            * (1 - self.coupling * self.infection2) ** l[3, x == 0]
        )

        # Node IS
        inf1[x == 1] = 1 - (1 - self.coupling * self.infection2) ** (
            l[2, x == 1] + l[3, x == 1]
        )

        # Node SI
        inf0[x == 2] = 1 - (1 - self.coupling * self.infection1) ** (
            l[1, x == 2] + l[3, x == 2]
        )
        return inf0, inf1

    def recovery(self, x, l):
        rec0 = np.ones(x.shape) * self.recovery1
        rec1 = np.ones(x.shape) * self.recovery2

        return rec0, rec1

    def number_of_infected(self, x):
        return x.size - np.sum(x == 0)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 3
        return x


class AsymmetricSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        SISSIS.__init__(self, config, **kwargs)
        boost = config.boost
        if boost == "source":
            self.infection = self._source_infection_
        elif boost == "target":
            self.infection = self._target_infection_
        else:
            raise ValueError(
                f"{boost} is invalid, valid entries are ['source', 'target']"
            )

    def _source_infection_(self, x, l):
        inf0 = np.zeros(x.shape)
        inf1 = np.zeros(x.shape)

        # Node SS
        inf0[x == 0] = 1 - (1 - self.infection1) ** (l[1, x == 0] + l[3, x == 0])
        inf1[x == 0] = 1 - (1 - self.infection2) ** (l[2, x == 0] + l[3, x == 0])

        # Node IS
        inf1[x == 1] = 1 - (1 - self.coupling * self.infection2) ** (
            l[2, x == 1] + l[3, x == 1]
        )

        # Node SI
        inf0[x == 2] = 1 - (1 - self.coupling * self.infection1) ** (
            l[1, x == 2] + l[3, x == 2]
        )
        return inf0, inf1

    def _target_infection_(self, x, l):
        inf0 = np.zeros(x.shape)
        inf1 = np.zeros(x.shape)

        # Node SS
        inf0[x == 0] = (
            1
            - (1 - self.infection1) ** l[1, x == 0]
            * (1 - self.coupling * self.infection1) ** l[3, x == 0]
        )
        inf1[x == 0] = (
            1
            - (1 - self.infection2) ** l[2, x == 0]
            * (1 - self.coupling * self.infection2) ** l[3, x == 0]
        )

        # Node IS
        inf1[x == 1] = (
            1
            - (1 - self.infection2) ** l[2, x == 1]
            * (1 - self.coupling * self.infection2) ** l[3, x == 1]
        )

        # Node SI
        inf0[x == 2] = (
            1
            - (1 - self.infection1) ** l[1, x == 2]
            * (1 - self.coupling * self.infection1) ** l[3, x == 2]
        )
        return inf0, inf1


class HiddenSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        SISSIS.__init__(self, config, **kwargs)
        self.state_map = {0: 0, 1: 1, 2: 0, 3: 1}
        self.hide = True

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        p = SISSIS.predict(self, x)

        if self.hide:
            ltp = np.zeros((x.shape[0], 2))
            ltp[:, 0] = p[:, 0] + p[:, 2]
            ltp[:, 1] = p[:, 1] + p[:, 3]
            return ltp

        return p

    def sample(self, x):
        p = SISSIS.predict(self, x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        x = np.array(dist.sample())
        return x


class PartiallyHiddenSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        SISSIS.__init__(self, config, **kwargs)
        self.state_map = {0: 0, 1: 1, 2: 0, 3: 1}
        self.hide_prob = config.hide_prob
        self.hide = True

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[-1]

        p = SISSIS.predict(self, x)

        if self.hide:
            ltp = np.zeros((x.shape[0], 4))
            ltp[:, 0] = p[:, 0] + self.hide_prob * p[:, 2]
            ltp[:, 1] = p[:, 1] + self.hide_prob * p[:, 3]
            ltp[:, 2] = (1 - self.hide_prob) * p[:, 2]
            ltp[:, 3] = (1 - self.hide_prob) * p[:, 3]
            return ltp

        return p

    def sample(self, x):
        p = SISSIS.predict(self, x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        x = np.array(dist.sample())
        return x

    def loglikelihood(self, x, real_x, y=None, real_y=None, g=None):
        if g is not None:
            self.network = g
        if y is None:
            y = np.roll(x, -1, axis=0)[:-1]
            x = x[:-1]
        if real_y is None:
            real_y = np.roll(real_x, -1, axis=0)[:-1]
            real_x = real_x[:-1]

        if x.shape == (self.window_size, self.num_nodes) or x.shape == (self.num_nodes):
            x = x.reshape(1, self.window_size, self.num_nodes)
            real_x = real_x.reshape(1, self.window_size, self.num_nodes)
            y = y.reshape(1, self.num_nodes)

        loglikelihood = 0
        for i in range(x.shape[0]):
            p = SISSIS.predict(self, real_x[i])
            onehot_y = onehot(real_y[i], num_class=self.num_states)
            p = (onehot_y * p).sum(-1)

            real_si = np.where(real_y[i] == 2)[0]
            real_ii = np.where(real_y[i] == 3)[0]
            si = np.where((y[i] == 2) * (real_y[i] == 2))[0]
            ii = np.where((y[i] == 3) * (real_y[i] == 3))[0]
            q = np.ones(self.num_nodes)
            q[real_si] = self.hide_prob
            q[real_ii] = self.hide_prob
            q[si] = 1 - self.hide_prob
            q[ii] = 1 - self.hide_prob
            p *= q

            p[p <= 1e-15] = 1e-15

            logp = np.log(p)
            loglikelihood += logp.sum()
        return loglikelihood


class SISnoise(StochasticEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        num_states = 4
        self.infection1 = config.infection1
        self.recovery = config.recovery
        self.noise = config.noise
        self.transform = RemapStateTransform()
        self.transform.state_map = {0: 0, 1: 1, 2: 0, 3: 1}

        StochasticEpidemics.__init__(self, config, num_states)

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        y = self.transform(x)
        ltp = np.zeros((x.shape[0], self.num_states))
        p = independent(self.neighbors_state(y)[1], self.infection1)
        q = self.recovery
        ltp[x == 0, 0] = (1 - p[x == 0]) * (1 - self.noise)
        ltp[x == 0, 1] = (p[x == 0]) * (1 - self.noise)
        ltp[x == 0, 2] = (1 - p[x == 0]) * self.noise
        ltp[x == 0, 3] = (p[x == 0]) * self.noise
        ltp[x == 1, 0] = (q) * (1 - self.noise)
        ltp[x == 1, 1] = (1 - q) * (1 - self.noise)
        ltp[x == 1, 2] = q * self.noise
        ltp[x == 1, 3] = (1 - q) * self.noise
        ltp[x == 2, 0] = (1 - p[x == 2]) * (1 - self.noise)
        ltp[x == 2, 1] = (p[x == 2]) * (1 - self.noise)
        ltp[x == 2, 2] = (1 - p[x == 2]) * self.noise
        ltp[x == 2, 3] = (p[x == 2]) * self.noise
        ltp[x == 1, 0] = (q) * (1 - self.noise)
        ltp[x == 1, 1] = (1 - q) * (1 - self.noise)
        ltp[x == 1, 2] = q * self.noise
        ltp[x == 1, 3] = (1 - q) * self.noise
        return ltp

    def number_of_infected(self, x):
        return x.size - np.sum(x == 0)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 3
        return x
