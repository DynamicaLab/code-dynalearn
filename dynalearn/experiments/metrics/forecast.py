import networkx as nx
import numpy as np

from abc import abstractmethod
from dynalearn.experiments.metrics import Metrics
from dynalearn.util import Verbose
from dynalearn.dynamics.trainable import VARDynamics


class ForecastMetrics(Metrics):
    def __init__(self, config):
        Metrics.__init__(self, config)
        self.num_steps = config.forecast.get("num_steps", [1])
        if isinstance(self.num_steps, int):
            self.num_steps = [self.num_steps]
        elif not isinstance(self.num_steps, list):
            self.num_steps = list(self.num_steps)

    @abstractmethod
    def get_model(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        return

    def compute(self, experiment, verbose=Verbose()):
        self.verbose = verbose
        self.initialize(experiment)

        self.model = self.get_model(experiment)
        datasets = {
            "total": experiment.dataset,
            "train": experiment.dataset,
            "val": experiment.val_dataset,
            "test": experiment.test_dataset,
        }
        datasets = {
            k: self._get_data_(v, total=k == "total")
            for k, v in datasets.items()
            if v is not None
        }
        nobs = [v.shape[0] for k, v in datasets.items()]
        self.num_updates = np.sum(
            [[s * (n - s + 1) for s in self.num_steps] for n in nobs]
        )
        pb = self.verbose.progress_bar(self.__class__.__name__, self.num_updates)
        for k, v in datasets.items():
            total = k == "total"
            for s in self.num_steps:
                self.data[f"{k}-{s}"] = self._get_forecast_(v, s, pb)

        if pb is not None:
            pb.close()

        self.exit(experiment)

    def _get_forecast_(self, dataset, num_steps=1, pb=None):
        if dataset.shape[0] - num_steps + 1 < 0:
            return np.zeros((0, *dataset.shape[1:-1]))
        y = np.zeros((dataset.shape[0] - num_steps + 1, *dataset.shape[1:-1]))
        for i, x in enumerate(dataset[:-num_steps]):
            for t in range(num_steps):
                yy = self.model.sample(x)
                x = np.roll(x, -1, axis=-1)
                x.T[-1] = yy.T
                if pb is not None:
                    pb.update()
            y[i] = yy
        return y

    def _get_data_(self, dataset, total=False):
        if dataset is None:
            return
        data = dataset.inputs[0].data
        if not total:
            w = dataset.state_weights[0].data
            data = dataset.inputs[0].data[w > 0]
        return data


class GNNForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        model = experiment.model
        model.network = experiment.dataset.networks[0].data
        return model


class TrueForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        model = experiment.dynamics
        model.network = experiment.dataset.networks[0].data
        return model


class VARForecastMetrics(ForecastMetrics):
    def get_model(self, experiment):
        model = VARDynamics(experiment.model.num_states, lag=experiment.model.lag)
        model.network = experiment.dataset.networks[0].data
        c = experiment.dataset.state_weights[0].data > 0
        X = experiment.dataset.inputs[0].data[c]
        Y = experiment.dataset.targets[0].data[c]
        model.fit(X, Y=Y)
        return model
