import h5py
import numpy as np
from abc import ABC, abstractmethod


class Data(ABC):
    def __init__(self, name="data"):
        self.name = name
        self._data = None

    def __eq__(self, other):
        if isinstance(other, Data):
            return self.data == other.data
        return False

    def copy(self):
        data_copy = self.__class__()
        data_copy.__dict__ = self.__dict__.copy()
        data_copy.data = self._data.copy()
        return data_copy

    # def transform(self, transformation):
    #     self.data = transformation(self.data)
    #     return self.data

    def get(self):
        return self.data

    def save(self, h5file):
        if self.name in h5file:
            del h5file[self.name]
        assert isinstance(self.data, np.ndarray)
        h5file.create_dataset(self.name, data=self.data)

    def load(self, h5file):
        if isinstance(h5file, h5py.Dataset):
            self.data = h5file[...]
        elif isinstance(h5file, h5py.Group):
            if self.name in h5file:
                self.data = h5file[self.name][...]
            else:
                print(
                    f"{self.name} not in h5file with name {h5file}. Available keys are {h5file.keys()}"
                )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data


class DataCollection:
    def __init__(self, name="data_collection", data_list=[], template=None):
        self.name = name
        self.data_list = []
        if template is None:
            self.template = lambda d: Data(data=d, shape=d.shape[1:])
        else:
            self.template = lambda d: template(data=d)
        for data in data_list:
            self.add(data)

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def __eq__(self, other):
        if isinstance(other, DataCollection):
            for d1, d2 in zip(self.data_list, other.data_list):
                if d1 != d2:
                    return False
        else:
            return False
        return True

    def add(self, x):
        assert issubclass(type(x), Data)
        x.name = "d" + str(len(self))
        self.data_list.append(x)

    def copy(self):
        data_copy = self.__class__()
        data_copy.__dict__ = self.__dict__.copy()
        data_copy.data_list = [data.copy() for data in self.data_list]
        return data_copy

    def save(self, h5file):
        assert isinstance(h5file, h5py.Group)
        group = h5file.create_group(self.name)
        for i, d in enumerate(self.data_list):
            d.save(group)

    def load(self, h5file):
        assert isinstance(h5file, h5py.Group)
        if self.name in h5file:
            group = h5file[self.name]
            for k, v in group.items():
                d = self.template(v)
                self.add(d)

    # def transform(self, transformation):
    #     for i, x in enumerate(self.data_list):
    #         self.data_list[i] = transformation(x)

    @property
    def size(self):
        return len(self)
