import numpy as np

from abc import abstractmethod
from random import sample
from dynalearn.util import poisson_distribution
from dynalearn.networks import ConfigurationNetworkGenerator, GNMNetworkGenerator
from dynalearn.config import NetworkConfig
from .metrics import Metrics
from .util import Initializer, ModelSampler, Statistics


class StationaryStateMetrics(Metrics):
    def __init__(self, config):
        Metrics.__init__(self, config)
        if "parameters" in config.__dict__:
            self.parameters = config.parameters
        else:
            self.parameters = None
        self.num_samples = config.num_samples
        self.initializer = Initializer(self.config)
        self.sampler = ModelSampler.getter(self.config)
        self.statistics = Statistics.getter(self.config)

        self.dynamics = None
        self.networks = None
        self.model = None

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplementedError()

    def get_networks(self, experiment):
        return experiment.networks

    def change_param(self, p):
        return

    def initialize(self, experiment):
        self.dynamics = experiment.dynamics
        self.networks = self.get_networks(experiment)
        self.model = self.get_model(experiment)
        self.initializer.setUp(self)
        self.sampler.setUp(self)

        if self.parameters is not None:
            factor = 0
            for k, p in self.parameters.items():
                factor += len(p)
                self.data[f"param-{k}"] = p
        else:
            factor = len(self.config.init_param)
        self.num_updates = self.num_samples * factor

        for m in self.initializer.all_modes:
            self.get_data[m] = lambda pb: self._all_stationary_states_(pb=pb)
            self.names.append(m)

    def initialize_network(self):
        g = self.networks.generate()
        self.dynamics.network = g
        self.model.network = self.dynamics.network

    def _stationary_(self, param=None, pb=None):
        if param is not None:
            self.change_param(param)
        samples = np.zeros((0, self.model.num_states))
        x0 = None
        for i in range(self.num_samples):
            self.initialize_network()
            if x0 is None:
                x0 = self.initializer()
            y, x0 = self.sampler(self.model, x0)
            samples = np.concatenate([samples, y])
            if pb is not None:
                pb.update()
        self.initializer.update(np.mean(samples, axis=0))
        y = self.statistics(samples)
        return y

    def _all_stationary_states_(self, pb=None):
        s = []
        mode = self.initializer.mode
        if self.parameters is not None:
            for p in self.parameters[mode]:
                s.append(self._stationary_(param=p, pb=pb))
        else:
            s.append(self._stationary_(pb=pb))
        self.initializer.next_mode()

        return np.array(s)


class TrueSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.dynamics


class GNNSSMetrics(StationaryStateMetrics):
    def get_model(self, experiment):
        return experiment.model


class PoissonSSMetrics(StationaryStateMetrics):
    def __init__(self, config):
        StationaryStateMetrics.__init__(self, config)
        self.num_nodes = config.num_nodes
        self.num_k = config.num_k

    def get_networks(self, experiment):
        p_k = poisson_distribution(self.parameters[0], self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        self.weights = experiment.networks.weights
        return ConfigurationNetwork(config, weights=self.weights)

    def change_param(self, avgk):
        p_k = poisson_distribution(avgk, self.num_k)
        config = NetworkConfig.configuration(self.num_nodes, p_k)
        self.networks = ConfigurationNetworkGenerator(config, weights=self.weights)


class TruePSSMetrics(TrueSSMetrics, PoissonSSMetrics):
    def __init__(self, config):
        TrueSSMetrics.__init__(self, config)
        PoissonSSMetrics.__init__(self, config)


class GNNPSSMetrics(GNNSSMetrics, PoissonSSMetrics):
    def __init__(self, config):
        GNNSSMetrics.__init__(self, config)
        PoissonSSMetrics.__init__(self, config)


class ErdosRenyiSSMetrics(StationaryStateMetrics):
    def __init__(self, config):
        StationaryStateMetrics.__init__(self, config)
        self.num_nodes = config.num_nodes

    def get_networks(self, experiment):
        m = 4.0 * self.num_nodes / 2
        config = NetworkConfig.gnm(self.num_nodes, m)
        self.weights = experiment.networks.weights
        return GNMNetworkGenerator(config, weights=self.weights)

    def change_param(self, avgk):
        self.networks.config.m = avgk * self.num_nodes / 2


class TrueERSSMetrics(TrueSSMetrics, ErdosRenyiSSMetrics):
    def __init__(self, config):
        TrueSSMetrics.__init__(self, config)
        ErdosRenyiSSMetrics.__init__(self, config)


class GNNERSSMetrics(GNNSSMetrics, ErdosRenyiSSMetrics):
    def __init__(self, config):
        GNNSSMetrics.__init__(self, config)
        ErdosRenyiSSMetrics.__init__(self, config)
