import copy
import h5py
import networkx as nx
import numpy as np
import tqdm

from itertools import islice, chain
from .sampler import Sampler
from scipy.stats import gaussian_kde

from dynalearn.config import Config
from dynalearn.datasets import TransformList
from dynalearn.datasets.data import (
    DataCollection,
    NetworkData,
    StateData,
)
from dynalearn.datasets.weights import (
    Weight,
    DegreeWeight,
    StrengthWeight,
)
from dynalearn.datasets.transforms.getter import get as get_transforms
from dynalearn.util import get_node_attr, Verbose, LoggerDict


class Dataset:
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        self.config = config
        self.bias = config.bias
        self.use_groundtruth = config.use_groundtruth
        self.sampler = Sampler(self)

        if "transforms" in config.__dict__:
            self.transforms = get_transforms(config.transforms)
        else:
            self.transforms = TransformList()

        self.use_transformed = len(self.transforms) > 0

        self._data = {}
        self._transformed_data = {}
        self._weights = None
        self._state_weights = None
        self._network_weights = None
        self._indices = None
        self._rev_indices = None

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return np.sum([s.size for s in self.data["inputs"].data_list])

    def __iter__(self):
        return self

    def __next__(self):
        return self[self.rev_indices[self.sampler()]]

    def generate(self, experiment, verbose=Verbose()):
        details = self.setup(experiment)
        self.transforms.setup(experiment)

        pb = verbose.progress_bar(
            "Generating training set", details.num_networks * details.num_samples
        )

        self.data = self._generate_data_(details, pb=pb)
        if self.use_groundtruth:
            self._data["ground_truth"] = self._generate_groundtruth_(self._data)
        if experiment.verbose == 1:
            pb.close()

    def partition(self, type="random", **kwargs):
        if type == "random":
            return self.random_partition(**kwargs)
        elif type == "cleancut":
            return self.cleancut_partition(**kwargs)
        else:
            raise ValueError(
                f"`{type}` is invalid, valid entries are `['random', 'cleancut']`."
            )

    def random_partition(self, fraction=0.1, bias=0, pb=None):
        dataset = type(self)(self.config)
        dataset._data = self._data
        if self.use_transformed:
            dataset._transformed_data = self._transformed_data
        weights = self.weights.copy()
        for i in range(self.networks.size):
            for j in range(self.inputs[i].size):
                index = np.where(self.weights[i].data[j] > 0)[0]
                n = np.random.binomial(self.weights[i].data.shape[-1], fraction)
                if n == 0:
                    n = 1
                if self.bias > 0:
                    p = self.weights[i].data[j, index] ** (-bias / self.bias)
                else:
                    p = self.weights[i].data[j, index]
                if p.sum() == 0:
                    continue
                else:
                    p /= p.sum()
                remove_nodes = np.random.choice(index, p=p, size=n, replace=False)
                weights[i].data[j] *= 0
                weights[i].data[j, remove_nodes] = (
                    self.weights[i].data[j, remove_nodes] * 1
                )
                self.weights[i].data[j, remove_nodes] = 0
                if pb is not None:
                    pb.update()

        dataset.weights = weights
        dataset.indices = self.indices
        dataset.lag = self.lag
        dataset.lagstep = self.lagstep
        dataset.maxlag = self.maxlag
        dataset.num_states = self.num_states
        dataset.sampler.reset()

        return dataset

    def cleancut_partition(self, ti=0, tf=-1):
        dataset = type(self)(self.config)
        dataset._data = self._data
        if self.use_transformed:
            dataset._transformed_data = self._transformed_data
        weights = self.weights.copy()
        new_weights = self.weights.copy()

        if isinstance(ti, int):
            ti = [ti] * self.networks.size

        if isinstance(tf, int):
            tf = [tf] * self.networks.size
        for i, _ti, _tf in zip(range(self.networks.size), ti, tf):
            if _ti == _tf:
                continue
            if _ti == -1:
                _ti = self.inputs[i].size
            if _tf == -1:
                _tf = self.inputs[i].size
            index = np.arange(_ti, _tf)
            c_index = np.concatenate(
                [np.arange(_ti), np.arange(_tf, self._data["inputs"][i].size)]
            )
            weights[i].data[_ti:_tf] = 0
            new_weights[i].data[:_ti] = 0
            new_weights[i].data[_tf:] = 0

        self.weights = weights
        dataset.weights = new_weights
        dataset.indices = self.indices
        dataset.lag = self.lag
        dataset.lagstep = self.lagstep
        dataset.maxlag = self.maxlag
        dataset.num_states = self.num_states
        dataset.sampler.reset()

        return dataset

    def setup(self, experiment):
        self.m_networks = experiment.networks
        self.m_dynamics = experiment.dynamics
        self.lag = experiment.model.lag
        self.lagstep = experiment.model.lagstep
        self.maxlag = experiment.train_details.maxlag
        self.num_states = experiment.model.num_states
        self.verbose = experiment.verbose
        return experiment.train_details

    def to_batch(self, size):
        sourceiter = iter(self)
        while True:
            if (
                self.sampler.counter < len(self)
                and len(self.sampler.avail_networks) > 0
            ):
                batchiter = islice(sourceiter, size)
                yield chain([next(batchiter)], batchiter)
            else:
                self.sampler.reset()
                return

    def save(self, h5file, name=None):
        assert isinstance(h5file, h5py.Group)
        if self.is_empty():
            return

        name = name or "data"
        if name in h5file:
            del h5file[name]
        group = h5file.create_group(name)
        self._save_data_(self._data, group)

        self.weights.save(group)

        if len(self._transformed_data) > 0:
            name = f"transformed_{name}"
            if name in h5file:
                del h5file[name]
            group = h5file.create_group(name)
            self._save_data_(self._transformed_data, h5file[name])

    def load(self, h5file):
        assert isinstance(h5file, h5py.Group)

        self._data = {}
        self._transformed_data = {}

        self._data = self._load_data_(h5file)

        w = Weight()
        w.load(h5file)
        self.weights = w

        if self.use_transformed:
            if "transformed_data" in h5file:
                self._transformed_data = self._load_data_("transformed_data", h5file)
            elif len(self._data) > 0:
                self._transformed_data = self._transform_data_(self._data)

    @property
    def data(self):
        if self.use_transformed:
            return self._transformed_data
        else:
            return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.use_transformed:
            self._transformed_data = self._transform_data_(data)
        self.weights = self._get_weights_()
        self.indices = self._get_indices_()
        self.sampler.reset()

    @property
    def inputs(self):
        return self.data["inputs"]

    @property
    def targets(self):
        if self.use_groundtruth and "ground_truth" in self._data:
            return self._data["ground_truth"]
        else:
            return self.data["targets"]

    @property
    def networks(self):
        return self.data["networks"]

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        assert isinstance(weights, DataCollection)
        self._weights = weights
        self._state_weights = weights.to_state_weights()
        self._network_weights = weights.to_network_weights()

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        self._indices = indices
        self._rev_indices = {(i, j): k for k, (i, j) in self._indices.items()}

    @property
    def rev_indices(self):
        return self._rev_indices

    @property
    def state_weights(self):
        return self._state_weights

    @property
    def network_weights(self):
        return self._network_weights

    def is_empty(self):
        if len(self.data) == 0:
            return True
        else:
            if (
                "inputs" not in self.data
                or "targets" not in self.data
                or "networks" not in self.data
            ):
                return True
            else:
                if (
                    self.data["inputs"].size == 0
                    or self.data["targets"].size == 0
                    or self.data["networks"].size == 0
                ):
                    return True
                else:
                    return False

    def _generate_data_(self, details, pb=None):
        networks = DataCollection(name="networks")
        inputs = DataCollection(name="inputs")
        targets = DataCollection(name="targets")
        back_step = (self.lag - 1) * self.lagstep

        for i in range(details.num_networks):
            g = self.m_networks.generate()
            self.m_dynamics.network = g
            x = self.m_dynamics.initial_state()

            networks.add(NetworkData(data=self.m_dynamics.network))

            in_data = np.zeros((*x.shape, self.lag))  # [nodes, features, lag]
            inputs_data = np.zeros((details.num_samples, *in_data.shape))
            targets_data = np.zeros((details.num_samples, *x.shape))
            t = 0
            j = 0
            k = 0
            while j < details.num_samples:
                if t % self.lagstep == 0:
                    in_data.T[k] = 1 * x.T
                    k += 1
                    if k == self.lag:
                        inputs_data[j] = 1 * in_data
                        for _ in range(self.lagstep):
                            y = self.m_dynamics.sample(x)
                            x = 1 * y
                        targets_data[j] = 1 * y

                        k = 0
                        j += 1
                        if pb is not None:
                            pb.update()
                        if j % details.resampling == 0 and details.resampling != -1:
                            x = self.m_dynamics.initial_state()
                    else:
                        x = self.m_dynamics.sample(x)
                else:
                    x = self.m_dynamics.sample(x)
                if details.resample_when_dead and self.m_dynamics.is_dead(x):
                    x = self.m_dynamics.initial_state()
                t += 1
            inputs.add(StateData(data=inputs_data))
            targets.add(StateData(data=targets_data))

        data = {
            "networks": networks,
            "inputs": inputs,
            "targets": targets,
        }

        return data

    def _generate_groundtruth_(self, data):
        ground_truth = DataCollection(name="ground_truth")
        for i in range(data["networks"].size):
            g = data["networks"][i].data
            self.m_dynamics.network = g
            num_samples = data["inputs"][i].size
            gt_data = None

            for j, x in enumerate(data["inputs"][i].data):
                y = self.m_dynamics.predict(x)
                if gt_data is None:
                    gt_data = np.zeros((num_samples, *y.shape))
                gt_data[j] = y
            ground_truth.add(StateData(data=gt_data))
        return ground_truth

    def _transform_data_(self, data):
        d = {}
        for k, v in data.items():
            vv = v.copy()
            for i in range(len(v)):
                vv[i].data = self.transforms(v[i]).data

            d[k] = v
        return d

    def _get_indices_(self):
        if self.data["inputs"] is None or self.data["networks"] is None:
            return {}
        index = 0
        indices_dict = {}
        for i in range(self.data["networks"].size):
            for j in range(self.data["inputs"][i].size):
                indices_dict[index] = (i, j)
                index += 1
        return indices_dict

    def _get_weights_(self):
        weights = Weight(bias=self.bias)
        weights.compute(self, verbose=self.verbose)
        return weights

    def _save_data_(self, data, h5file):
        data["networks"].save(h5file)
        data["inputs"].save(h5file)
        data["targets"].save(h5file)
        if "ground_truth" in data:
            data["ground_truth"].save(h5file)

    def _load_data_(self, h5file):
        data = {
            "networks": DataCollection(name="networks", template=NetworkData),
            "inputs": DataCollection(name="inputs", template=StateData),
            "targets": DataCollection(name="targets", template=StateData),
            "ground_truth": DataCollection(name="ground_truth", template=StateData),
        }

        for d_type in ["networks", "inputs", "targets", "ground_truth"]:
            data[d_type].load(h5file)
        return data


class StructureWeightDataset(Dataset):
    def _get_weights_(self):
        if self.config.use_strength:
            weights = StrengthWeight()
        else:
            weights = DegreeWeight()
        weights.compute(self, verbose=self.verbose)
        return weights
