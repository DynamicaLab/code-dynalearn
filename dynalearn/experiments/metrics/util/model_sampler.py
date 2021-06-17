import numpy as np

from scipy.signal import savgol_filter
from dynalearn.util import from_nary


class ModelSampler(ABC):
    def __init__(self, config):
        self.config = config

    def __call__(self, mode, initializer, statistics):
        raise NotImplementedError()

    def aggregate(self, x):
        raise NotImplementedError()

    def setUp(self, metrics):
        self.dynamics = metrics.dynamics
        self.num_states = metrics.model.num_states
        self.lag = metrics.model.lag
        self.lagstep = metrics.model.lagstep

    def burning(self, model, x, burn=1):
        for b in range(burn):
            y = x.T[:: self.lagstep]
            y = model.sample(y.T)
            x = np.roll(x, -1, axis=-1)
            x.T[-1] = y.T
        return x

    @classmethod
    def getter(cls, config):
        __all_samplers__ = {
            "SteadyStateSampler": SteadyStateSampler,
            "FixedPointSampler": FixedPointSampler,
        }

        if config.sampler in __all_samplers__:
            return __all_samplers__[config.sampler](config)
        else:
            raise ValueError(
                f"`{config.sampler}` is invalid, valid entries are `{__all_samplers__.key()}`"
            )


class SteadyStateSampler(ModelSampler):
    def __init__(self, config):
        ModelSampler.__init__(self, config)
        self.T = config.T
        self.burn = config.burn
        self.tol = config.tol

    def __call__(self, model, x0):
        x, x0 = self.collect(model, x0, self.T)
        index = self.where_steady(x[:, 0])
        dt = self.T - index
        if dt < self.tol:
            y, x0 = self.collect(model, x0, self.tol - dt)
            x = np.concatenate([x[index:], y])
        return x, x0

    def collect(self, model, x0, T):
        x = np.zeros((self.T, model.num_states))
        for t in range(self.T):
            x0 = self.burning(model, x0, self.burn)
            x[t] = self.aggregate(x0)
            if self.dynamics.is_dead(x0):
                x0 = self.nearly_dead_state()
        return x, x0

    def where_steady(self, x, window=51, polyorder=2, tol=1e-4, maxiter=50):
        filtered = savgol_filter(x, window, polyorder)
        index = 0
        if filtered.mean() / filtered.std() < 5:
            derivative = np.gradient(filtered, 1) ** 2
            max_index = np.argmax(derivative)
            derivative = derivative[max_index:]
            diff = np.inf
            it = 0
            while diff > tol and it < maxiter:
                epsilon = derivative[index:].mean()
                index = np.where(derivative < epsilon)[0]
                if len(index) > 0:
                    index = index.min()
                else:
                    index = 0
                diff = (epsilon - derivative[index:].mean()) / derivative[index:].mean()
                it += 1
            index += max_index
        return index

    def aggregate(self, x):
        agg_x = []
        assert x.shape[-1] == self.lag * self.lagstep
        if x.ndim == 2:
            x = from_nary(x[:, :: self.lagstep], axis=-1, base=self.num_states)
        elif x.ndim == 3:
            x = np.array(
                [
                    from_nary(xx[:, :: self.lagstep], axis=-1, base=self.num_states)
                    for xx in x
                ]
            )
        agg_x = np.array(
            [np.mean(x == i, axis=-1) for i in range(self.num_states ** self.lag)]
        )
        return agg_x

    def nearly_dead_state(self):
        x0 = self.dynamics.nearly_dead_state()
        x0 = np.repeat(np.expand_dims(x0, 1), self.lag * self.lagstep, axis=1)
        return x0


class FixedPointSampler(ModelSampler):
    def __init__(self, config):
        ModelSampler.__init__(self, config)
        self.initial_burn = config.initial_burn
        self.mid_burn = config.mid_burn
        self.tol = config.tol
        self.maxiter = config.maxiter

    def __call__(self, model, x0):
        x = self.burning(model, x0, self.initial_burn)

        diff = np.inf
        i = 0
        while diff > self.tol and i < self.maxiter:
            y = self.burning(model, x, self.mid_burn)
            diff = self.distance(x, y)
            x = y * 1
            i += 1
        agg_y = np.expand_dims(self.aggregate(y), 0)
        return agg_y, None

    def distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def aggregate(self, x):
        assert x.shape[-1] == self.lag * self.lagstep
        assert x.shape[-2] == self.num_states
        return x.mean((0, -1))
