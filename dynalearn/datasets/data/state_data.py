import h5py
import numpy as np
from abc import ABC, abstractmethod

from dynalearn.datasets.data.data import Data


class StateData(Data):
    def __init__(self, name="state_data", data=None):
        Data.__init__(self, name=name)
        if data is not None:
            if isinstance(data, h5py.Dataset):
                data = data[...]
            assert isinstance(data, np.ndarray)
            self.data = data

    def __eq__(self, other):
        if isinstance(other, StateData):
            return np.all(self.data == other.data)
        return False

    def get(self, index):
        return self._data[index]

    @property
    def size(self):
        if len(self._data.shape) > 1:
            return self._data.shape[0]
        else:
            return 1

    @property
    def shape(self):
        if len(self._data.shape) > 1:
            return self._data.shape[1:]
        else:
            return (0,)
