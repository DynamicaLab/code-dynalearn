import numpy as np

from ..dynamics import Dynamics, MultiplexDynamics
from dynalearn.config import Config
from scipy.stats import multivariate_normal


class SimpleVARDynamics(Dynamics):
    def __init__(self, num_states, config=None, **kwargs):
        config = Config(**kwargs) if config is None else config
        Dynamics.__init__(self, config, num_states)
        p = kwargs.copy()
        self.lag = kwargs.pop("lag", 1)
        self.varparams = None
        self.coefs = None
        self.intercept = None
        self.cov = None
        self._data = None
        self.dist = None

    @property
    def data(self):
        assert self._data is not None
        return self._data

    @property
    def nfeatures(self):
        return self.num_nodes * self.num_states

    def fit(self, X, Y=None):
        self._num_nodes = X.shape[1]
        self._num_states = X.shape[2]
        if Y is None:
            assert X.ndim == 3
            x = np.array([X[t - self.lag : t].ravel() for t in range(self.lag, len(X))])
            y = X[self.lag :].reshape(x.shape[0], -1)
            self._data = X
        else:
            assert X.ndim == 4
            assert X.shape[0] == Y.shape[0]
            self.lag = X.shape[-1]
            x = np.transpose(X, (0, 3, 1, 2))
            x = x.reshape(X.shape[0], -1)
            y = Y.reshape(X.shape[0], -1)
            self._data = (X, Y)
        nobs = x.shape[0]
        x = np.column_stack((np.ones(nobs), x))
        self.varparams = np.linalg.lstsq(x, y, rcond=1e-15)[0]
        self.coefs = self.varparams[1:].reshape(
            self.lag, self.nfeatures, self.nfeatures
        )
        self.intercept = self.varparams[0]
        residu = y - np.dot(x, self.varparams)
        self.cov = np.dot(residu.T, residu) / (nobs - self.nfeatures * self.lag - 1)

    def predict(self, x):
        assert x.ndim == 3, f"`ndim` is invalid, expected 3 but received {x.ndim}."
        assert x.shape == (
            self.num_nodes,
            self.num_states,
            self.lag,
        ), f"`shape` is invalid, expected {(self.num_nodes, self.num_states, self.lag)} but received {x.shape}."
        x = x.reshape(-1, self.lag).T
        y = np.dot(x.flatten(), self.coefs.reshape(-1, self.nfeatures)) + self.intercept
        y = y.reshape(self.num_nodes, self.num_states)
        return y

    def sample(self, x):
        assert x.ndim == 3, f"`ndim` is invalid, expected 3 but received {x.ndim}."
        assert x.shape == (self.num_nodes, self.num_states, self.lag)
        y = self.predict(x)
        mean = y.reshape(-1)
        s = multivariate_normal.rvs(mean, self.cov)
        s = s.reshape(self.num_nodes, self.num_states)
        return s

    def loglikelihood(self, X):
        assert X.ndim == 3
        x = np.array([X[t - self.lag : t] for t in range(self.lag, len(X))])
        x = np.transpose(x, (0, 2, 3, 1))
        y = X[self.lag :].reshape(x.shape[0], -1)
        logp = 0
        for i, (xx, yy) in enumerate(zip(x, y)):
            m = self.predict(xx).reshape(-1)
            logp += multivariate_normal.logpdf(yy.reshape(-1), mean=m, cov=self.cov)
        return logp

    def initial_state(self):
        if isinstance(self.data, tuple):
            nobs = self.data[0].shape[0]
            t = np.random.choice(range(nobs))
            x = self.data[0][t]
        elif isinstance(self.data, np.ndarray):
            nobs = self.data.shape[0]
            t = np.random.choice(range(self.lag, nobs))
            x = self.data[t - self.lag : t]
            x = np.transpose(x, (1, 2, 0))
        return x

    def is_dead(self, x):
        if np.any(np.isnan(x)):
            return True
        else:
            return False


class MultiplexVARDynamics(MultiplexDynamics, SimpleVARDynamics):
    def __init__(self, num_states, config=None, **kwargs):
        config = Config(**kwargs) if config is None else config
        SimpleVARDynamics.__init__(self, num_states, config=config, **kwargs)
        MultiplexDynamics.__init__(self, config, num_states)


def VARDynamics(num_states, config=None, **kwargs):
    if config is None:
        config = Config(**kwargs)

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False
    if is_multiplex:
        return MultiplexVARDynamics(num_states, config=config, **kwargs)
    else:
        return SimpleVARDynamics(num_states, config=config, **kwargs)
