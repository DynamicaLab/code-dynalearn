import networkx as nx
import numpy as np

from .base import StochasticEpidemics
from dynalearn.config import Config
from dynalearn.dynamics.activation import constant, threshold, nonlinear, sine, planck


class ComplexSIS(StochasticEpidemics):
    def __init__(self, config, activation, deactivation):
        StochasticEpidemics.__init__(self, config, 2)
        self.activation = activation
        self.deactivation = deactivation

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        ltp = np.zeros((*x.shape, self.num_states))
        l = self.neighbors_state(x)
        p = self.activation(l)
        q = self.deactivation(l)
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 1, 0] = q[x == 1]
        ltp[x == 1, 1] = 1 - q[x == 1]
        return ltp

    def number_of_infected(self, x):
        return np.sum(x == 1)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 1
        return x


class ComplexSIR(StochasticEpidemics):
    def __init__(self, config, activation, deactivation):
        StochasticEpidemics.__init__(self, config, 2)
        self.activation = activation
        self.deactivation = deactivation

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[:, -1].squeeze()
        ltp = np.zeros((x.shape[0], self.num_states))
        l = self.neighbors_state(x)
        p = self.activation(l)
        q = self.deactivation(l)
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 0, 2] = 0
        ltp[x == 1, 0] = 0
        ltp[x == 1, 1] = 1 - q[x == 1]
        ltp[x == 1, 2] = q[x == 1]
        ltp[x == 2, 0] = 0
        ltp[x == 2, 1] = 0
        ltp[x == 2, 2] = 1
        return ltp

    def number_of_infected(self, x):
        return np.sum(x == 1)

    def nearly_dead_state(self, num_infected=None):
        num_infected = num_infected or 1
        x = np.zeros(self.num_nodes)
        i = np.random.choice(range(self.num_nodes), size=num_infected)
        x[i] = 1
        return x


class ThresholdSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: threshold(l[1], l.sum(0), config.threshold, config.slope)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(ThresholdSIS, self).__init__(config, activation, deactivation)


class ThresholdSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: threshold(l[1], l.sum(0), config.threshold, config.slope)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(ThresholdSIR, self).__init__(config, activation, deactivation)


class NonLinearSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: nonlinear(l[1], config.infection, config.exponent)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(NonLinearSIS, self).__init__(config, activation, deactivation)


class NonLinearSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: nonlinear(l[1], config.infection, config.exponent)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(NonLinearSIR, self).__init__(config, activation, deactivation)


class SineSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: sine(
            l[1], config.infection, config.amplitude, config.period
        )
        deactivation = lambda l: constant(l[0], config.recovery)

        super(SineSIS, self).__init__(config, activation, deactivation)


class SineSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: sine(
            l[1], config.infection, config.amplitude, config.period
        )
        deactivation = lambda l: constant(l[0], config.recovery)

        super(SineSIR, self).__init__(config, activation, deactivation)


class PlanckSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: planck(l[1], config.temperature)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(PlanckSIS, self).__init__(config, activation, deactivation)


class PlanckSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)

        activation = lambda l: planck(l[1], config.temperature)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(PlanckSIR, self).__init__(config, activation, deactivation)
