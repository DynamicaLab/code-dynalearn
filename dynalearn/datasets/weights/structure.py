import numpy as np

from scipy.stats import gmean
from .weight import Weight
from .kde import KernelDensityEstimator


class DegreeWeight(Weight):
    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [
                dataset.networks[i].data.number_of_nodes()
                for i in range(dataset.networks.size)
            ]
        )

    def _get_features_(self, network, states, pb=None):
        for k in network.degree():
            self._add_features_(k)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))

        z = sum(self.features.values())
        for i, k in enumerate(network.degree()):
            weights[:, i] = self.features[k] / z
            if pb is not None:
                pb.update()
        return weights


class StrengthWeight(Weight):
    def setUp(self, dataset):
        self.num_updates = 2 * np.sum(
            [
                dataset.networks[i].data.number_of_nodes()
                for i in range(dataset.networks.size)
            ]
        )

    def _get_features_(self, network, states, pb=None):
        for i, k in enumerate(network.degree()):
            self._add_features_(("degree", k))
            for j in network.neighbors(i):
                if "weight" in network.data.edges[i, j]:
                    ew = network.data.edges[i, j]["weight"]
                else:
                    ew = 1
                self._add_features_(("weight", k), ew)
            if pb is not None:
                pb.update()

    def _get_weights_(self, network, states, pb=None):
        weights = np.zeros((states.shape[0], states.shape[1]))

        z = 0
        kde = {}
        mean = {}
        std = {}
        for k, v in self.features.items():
            if k[0] == "degree":
                z += v
            elif k[0] == "weight":
                kde[k[1]] = KernelDensityEstimator(samples=v)
        for i, k in enumerate(network.degree()):
            ew = []
            for j in network.neighbors(i):
                if "weight" in network.data.edges[i, j]:
                    ew.append(network.data.edges[i, j]["weight"])
                else:
                    ew.append(1)
            if k > 0:
                p = gmean(kde[k].pdf(ew))
            else:
                p = 1.0
            weights[:, i] = self.features[("degree", k)] / z * p

            if pb is not None:
                pb.update()
        return weights
