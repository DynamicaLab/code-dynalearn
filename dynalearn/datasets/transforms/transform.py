import networkx as nx
import numpy as np

from abc import abstractmethod
from dynalearn.config import Config
from dynalearn.datasets.data import Data, StateData, NetworkData


class Transform:
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.config = config

    def setup(self, experiment):
        return

    @abstractmethod
    def __call__(self, x):
        raise NotImplemented()


class TransformList(Transform):
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __len__(self):
        return len(self.transforms)

    def setup(self, experiment):
        for t in self.transforms:
            t.setup(experiment)


class StateTransform(Transform):
    @abstractmethod
    def _transform_state_(self, x):
        raise NotImplemented()

    def __call__(self, x):
        if not issubclass(type(x), StateData):
            return x
        data = x.data
        assert isinstance(data, np.ndarray)
        x.data = self._transform_state_(data)
        return x


class NetworkTransform(Transform):
    @abstractmethod
    def _transform_network_(self, g):
        raise NotImplemented()

    def __call__(self, x):
        if not issubclass(type(x), NetworkData):
            return x
        g = x.data
        x.data = self._transform_network_(g)
        return x
