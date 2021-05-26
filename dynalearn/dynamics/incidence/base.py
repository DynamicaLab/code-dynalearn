import networkx as nx
import numpy as np

from abc import abstractmethod
from itertools import product
from dynalearn.dynamics.dynamics import (
    Dynamics,
    WeightedDynamics,
    MultiplexDynamics,
    WeightedMultiplexDynamics,
)
from dynalearn.nn.models import Propagator
from dynalearn.networks import Network, MultiplexNetwork


class IncidenceEpidemics(Dynamics):
    def __init__(self, config):
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
        Dynamics.__init__(self, config, 1)

    @abstractmethod
    def predict(self, x):
        raise NotImplemented()

    def propagate(self, x):
        k = self.node_degree
        k[k == 0] = 1
        update = self.propagator(x, self.edge_index).cpu().detach().numpy().squeeze()
        update[k == 0] = 0
        k[k == 0] = 1
        return update

    def initial_state(self, init_param=None, density=None, squeeze=True):
        if init_param is None:
            init_param = self.init_param
        if not isinstance(init_param, float):
            p = np.random.rand()
        else:
            assert 0 < init_param < 1
            p = ini_param

        x = np.zeros([self.num_nodes, 1])
        self.population = self.init_population(density=density)

        for i, n in enumerate(self.population):
            x[i] = np.random.binomial(n, p)

        x = np.concatenate(
            [np.zeros((*x.shape, self.lag - 1)), x.reshape(*x.shape, 1)], axis=-1
        )
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

    def sample(self, x):
        return self.predict(x)

    def is_dead(self, x):
        return False

    def update_node_attr(self):
        node_attr = self.network.node_attr
        if "population" in node_attr:
            self.population = node_attr["population"]
        else:
            self.population = self.init_population()
            self.network.node_attr = {"population": self.population.copy()}


class WeightedIncidenceEpidemics(IncidenceEpidemics, WeightedDynamics):
    def __init__(self, config):
        IncidenceEpidemics.__init__(self, config)
        WeightedDynamics.__init__(self, config, 1)

    def propagate(self, x):
        k = self.node_degree
        s = self.node_strength
        s[s == 0] = 1
        update = (
            self.propagator(x, self.edge_index, w=self.edge_weight)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        update[s == 0] = 0
        update[k == 0] = 0
        return update * k / s


class MultiplexIncidenceEpidemics(IncidenceEpidemics, MultiplexDynamics):
    def __init__(self, config):
        IncidenceEpidemics.__init__(self, config)
        MultiplexDynamics.__init__(self, config, 1)

    def propagate(self, x):
        k = self.node_degree
        k[k == 0] = 1
        inf_update = (
            self.propagator(x, self.collapsed_network.edges)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        inf_update[k == 0] = 0
        return inf_update


class WeightedMultiplexIncidenceEpidemics(
    IncidenceEpidemics, WeightedMultiplexDynamics
):
    def __init__(self, config):
        IncidenceEpidemics.__init__(self, config)
        WeightedMultiplexDynamics.__init__(self, config, 1)

    def propagate(self, x):
        edges = self.collapsed_network.edges.T
        k = self.collapsed_node_degree
        s = self.collapsed_node_strength
        w = self.collapsed_edge_weight
        inf_update = (
            self.propagator(x, self.collapsed_edge_index, w=self.collapsed_edge_weight)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
        )
        inf_update[s == 0] = 0
        inf_update[k == 0] = 0
        s[s == 0] = 1
        return inf_update * k / s
