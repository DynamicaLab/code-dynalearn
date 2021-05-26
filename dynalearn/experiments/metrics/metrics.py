import h5py
import numpy as np
import tqdm

from abc import ABC, abstractmethod
from dynalearn.util import Verbose


class Metrics(ABC):
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.names = []
        self.get_data = {}
        self.num_updates = 0

    @abstractmethod
    def initialize(self, experiment):
        raise NotImplementedError("initialize must be implemented.")

    def exit(self, experiment):
        return

    def compute(self, experiment, verbose=Verbose()):
        self.verbose = verbose
        self.initialize(experiment)

        pb = self.verbose.progress_bar(self.__class__.__name__, self.num_updates)
        for k in self.names:
            d = self.get_data[k](pb=pb)
            if isinstance(d, dict):
                for kk, vv in d.items():
                    self.data[k + "/" + kk] = vv

            elif isinstance(d, (float, int, np.ndarray)):
                self.data[k] = d

        if pb is not None:
            pb.close()

        self.exit(experiment)

    def update(self, data):
        self.data.update(data)

    def save(self, h5file, name=None):
        if not isinstance(h5file, (h5py.File, h5py.Group)):
            raise ValueError("Dataset file format must be HDF5.")

        name = name or self.__class__.__name__

        for k, v in self.data.items():
            path = name + "/" + str(k)
            if path in h5file:
                del h5file[path]
            h5file.create_dataset(path, data=v)

    def load(self, h5file, name=None):
        if not isinstance(h5file, (h5py.File, h5py.Group)):
            raise ValueError("Dataset file format must be HDF5.")

        name = name or self.__class__.__name__

        if name in h5file:
            self.data = self.read_h5_recursively(h5file[name])

    def read_h5_recursively(self, h5file, prefix=""):
        ans_dict = {}
        for key in h5file:
            item = h5file[key]
            if prefix == "":
                path = f"{key}"
            else:
                path = f"{prefix}/{key}"

            if isinstance(item, h5py.Dataset):
                ans_dict[path] = item[...]
            elif isinstance(item, h5py.Group):
                d = self.read_h5_recursively(item, path)
                ans_dict.update(d)
            else:
                raise ValueError()
        return ans_dict


class CustomMetrics(Metrics):
    def initialize(self, experiment):
        return
