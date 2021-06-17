import networkx as nx
import numpy as np

from itertools import product
from dynalearn.dynamics.dynamics import (
    Dynamics,
    WeightedDynamics,
    MultiplexDynamics,
    WeightedMultiplexDynamics,
)
from dynalearn.nn.models import Propagator
from dynalearn.networks import Network, MultiplexNetwork


class DeterministicEpidemics(Dynamics):
    def __init__(self, config, num_states):
        if "init_param" in config.__dict__:
            self.init_param = config.init_param
        else:
            self.init_param = -1
        if "density" in config.__dict__:
            self.density = config.density
        else:
            self.density = -1

        self.population = None
        self.propagator = Propagator()
        Dynamics.__init__(self, config, num_states)

    def update(self, x):
        raise NotImplementedError()

    def infection_rate(self, x):
        raise NotImplementedError()

    def infection(self, x):
        infection = self.infection_rate(x).squeeze()
        k = self.node_degree
        k[k == 0] = 1
        update = (
            self.propagator(infection, self.edge_index).cpu().detach().numpy().squeeze()
        )
        update[k == 0] = 0
        k[k == 0] = 1
        return update

    def initial_state(self, init_param=None, density=None, squeeze=True):
        if init_param is None:
            init_param = self.init_param
        if not isinstance(init_param, (np.ndarray, list)):
            p = np.random.rand(self.num_states)
            p /= p.sum()
        else:
            assert len(init_param) == self.num_states
            p = np.array(init_param)

        x = np.zeros([self.num_nodes, self.num_states])
        self.population = self.init_population(density=density)

        for i, n in enumerate(self.population):
            x[i] = np.random.multinomial(n, p) / n

        x = x.reshape(*x.shape, 1).repeat(self.lag, -1)
        if squeeze:
            return x.squeeze()
        else:
            return x

    def init_population(self, density=None):
        if density is None:
            density = self.density
        if density == -1.0:
            density = self.num_nodes
        if "population" in self.network.node_attr:
            population = self.network.node_attr["population"]
        else:
            if isinstance(density, (float, int)):
                population = np.random.poisson(density, size=self.num_nodes)
            elif isinstance(density, (list, np.ndarray)):
                assert len(density) == self.num_nodes
                population = np.array(density)
        self.network.node_attr["population"] = population
        population[population <= 0] = 1
        return population

    def loglikelihood(self, x):
        return 1

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        dx = self.update(x)
        y = x + dx
        y[y < 0] = 0
        y[y > 1] = 1
        y /= y.sum(-1, keepdims=True)

        return y

    def sample(self, x):
        return self.predict(x)

    def is_dead(self, x):
        if np.all(x[:, 1] == 0):
            return True
        else:
            return False

    def update_node_attr(self):
        node_attr = self.network.node_attr
        if "population" in node_attr:
            self.population = node_attr["population"]
        else:
            self.population = self.init_population()
            self.network.node_attr = {"population": self.population.copy()}


class WeightedDeterministicEpidemics(DeterministicEpidemics, WeightedDynamics):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        WeightedDynamics.__init__(self, config, num_states)

    def infection(self, x):
        infection = self.infection_rate(x).squeeze()
        k = self.node_degree
        s = self.node_strength
        s[s == 0] = 1
        update = (
            self.propagator(infection, self.edge_index, w=self.edge_weight)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        update[s == 0] = 0
        update[k == 0] = 0
        s[s == 0] = 1
        return update * k / s


class MultiplexDeterministicEpidemics(DeterministicEpidemics, MultiplexDynamics):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        MultiplexDynamics.__init__(self, config, num_states)

    def infection(self, x):
        infection = self.infection_rate(x).squeeze()
        k = self.node_degree
        k[k == 0] = 1
        inf_update = (
            self.propagator(infection, self.collapsed_network.edges)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        inf_update[k == 0] = 0
        return inf_update


class WeightedMultiplexDeterministicEpidemics(
    DeterministicEpidemics, WeightedMultiplexDynamics
):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        WeightedMultiplexDynamics.__init__(self, config, num_states)

    def infection(self, x):
        infection = self.infection_rate(x).squeeze()
        edges = self.collapsed_network.edges.T
        k = self.collapsed_node_degree
        s = self.collapsed_node_strength
        w = self.collapsed_edge_weight
        inf_update = (
            self.propagator(
                infection, self.collapsed_edge_index, w=self.collapsed_edge_weight
            )
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        inf_update[s == 0] = 0
        inf_update[k == 0] = 0
        s[s == 0] = 1
        return inf_update * k / s
