import numpy as np


class Initializer:
    def __init__(self, config):
        self.config = config
        self._init_param = config.init_param
        self.current_param = self._init_param.copy()
        self.init_epsilon = config.init_epsilon
        self.adaptive = config.adaptive
        self.all_modes = list(self._init_param.keys())
        self.num_modes = len(self._init_param.keys())
        self._mode = self.all_modes[0]
        self.mode_index = 0

    def __call__(self):
        _x0 = self.dynamics.initial_state(init_param=self.current_param[self.mode])
        x0 = np.zeros((*_x0.shape, self.lag * self.lagstep))
        x0.T[0] = _x0.T
        for i in range(1, self.lag * self.lagstep):
            x0.T[i] = self.dynamics.sample(x0[i - 1]).T
        return x0

    def setUp(self, metrics):
        self.dynamics = metrics.dynamics
        self.num_states = metrics.model.num_states
        self.lag = metrics.model.lag
        self.lagstep = metrics.model.lagstep

    def update(self, x):
        assert x.shape == (self.num_states,)
        if self.adaptive:
            x[x < self.init_epsilon] = self.init_epsilon
            self.current_param[self.mode] = x
            self.current_param[self.mode] /= self.current_param[self.mode].sum()

    def next_mode(self):
        self.mode_index += 1
        if self.mode_index == len(self.all_modes):
            self.mode_index = 0
        self.mode = self.all_modes[self.mode_index]
        return self.mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode in self.all_modes:
            self._mode = mode
        else:
            raise ValueError(
                f"Mode `{mode}` is not invalid, available modes are `{self.all_modes}`"
            )
