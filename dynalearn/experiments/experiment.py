import h5py
import json
import networkx as nx
import numpy as np
import os
import pickle
import random
import shutil
import torch
import tqdm
import zipfile

from datetime import datetime
from dynalearn.datasets.getter import get as get_dataset
from dynalearn.dynamics.getter import get as get_dynamics
from dynalearn.experiments.metrics.getter import get as get_metrics
from dynalearn.networks.getter import get as get_network
from dynalearn.nn.metrics import get as get_train_metrics
from dynalearn.nn.callbacks.getter import get as get_callbacks
from dynalearn.util.loggers import (
    LoggerDict,
    MemoryLogger,
    TimeLogger,
)
from dynalearn.util import Verbose
from os.path import join, exists


class Experiment:
    def __init__(self, config, verbose=Verbose()):
        self.config = config
        self.name = config.name

        # Main objects
        if "modes" not in config.dataset.__dict__:
            config.dataset.modes = ["main"]
        self.all_modes = config.dataset.modes
        assert "main" in self.all_modes
        self._mode = "main"
        self._dataset = {k: get_dataset(config.dataset) for k in self.all_modes}
        self._val_dataset = {}
        self._test_dataset = {}

        self.networks = get_network(config.networks)
        self.dynamics = get_dynamics(config.dynamics)
        self.model = get_dynamics(config.model)
        self.config.model.num_params = int(self.model.nn.num_parameters())

        # Training related
        self.train_details = config.train_details
        self.metrics = get_metrics(config.metrics)
        self.train_metrics = get_train_metrics(config.train_metrics)
        self.callbacks = get_callbacks(config.callbacks)

        # Files location
        self.path_to_data = config.path_to_data
        self.path_to_best = config.path_to_best
        self.path_to_summary = config.path_to_summary

        # File names
        self.fname_data = (
            config.fname_data if "fname_data" in config.__dict__ else "data.h5"
        )
        self.fname_model = (
            config.fname_model if "fname_model" in config.__dict__ else "model.pt"
        )
        self.fname_optim = (
            config.fname_optim if "fname_optim" in config.__dict__ else "optim.pt"
        )
        self.fname_metrics = (
            config.fname_metrics if "fname_metrics" in config.__dict__ else "metrics.h5"
        )
        self.fname_history = (
            config.fname_history
            if "fname_history" in config.__dict__
            else "history.pickle"
        )
        self.fname_config = (
            config.fname_config
            if "fname_config" in config.__dict__
            else "config.pickle"
        )
        self.fname_logger = (
            config.fname_logger if "fname_logger" in config.__dict__ else "log.json"
        )

        # Setting verbose
        if isinstance(verbose, int):
            if verbose == 1 or verbose == 2:
                self.verbose = Verbose(
                    filename=join(self.path_to_data, "verbose"), vtype=verbose
                )
            else:
                self.verbose = Verbose(type=verbose)
        elif isinstance(verbose, Verbose):
            self.verbose = verbose

        # Setting seeds
        if "seed" in config.__dict__:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.__tasks__ = [
            "load",
            "save",
            "generate_data",
            "partition_val_dataset",
            "partition_test_dataset",
            "train_model",
            "compute_metrics",
            "zip",
        ]
        self.loggers = LoggerDict({"time": TimeLogger(), "memory": MemoryLogger()})
        self.__files__ = [
            "config.pickle",
            "log.json",
            "data.h5",
            "metrics.h5",
            "history.pickle",
            "model.pt",
            "optim.pt",
        ]

    # Run command
    def run(self, tasks=None):
        tasks = tasks or self.__tasks__

        self.begin()
        self.save_config()
        for t in tasks:
            if t in self.__tasks__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.__tasks__}`"
                )

        self.end()

    def begin(self):
        self.loggers.on_task_begin()
        self.verbose(f"---Experiment {self.name}---")
        if "time" in self.loggers.keys():
            begin = self.loggers["time"].log["begin"]
            self.verbose(f"Current time: {begin}")
        self.verbose(f"\n---Config---")
        self.verbose(f"{self.config}")

    def end(self):
        self.loggers.on_task_end()
        self.verbose(f"\n---Finished {self.name}---")
        if "time" in self.loggers.keys():
            end = self.loggers["time"].log["end"]
            t = self.loggers["time"].log["time"]
            self.verbose(f"Current time: {end}")
            self.verbose(f"Computation time: {t}\n")

    # All tasks
    def train_model(self, save=True, restore_best=True):
        self.verbose("\n---Training model---")

        self.model.nn.fit(
            self.dataset,
            epochs=self.train_details.epochs,
            batch_size=self.train_details.batch_size,
            val_dataset=self.val_dataset,
            metrics=self.train_metrics,
            callbacks=self.callbacks,
            loggers=self.loggers,
            verbose=self.verbose,
        )

        if save:
            self.save_model()

        if restore_best:
            self.load_model()

    def generate_data(self, save=True):
        self.verbose("\n---Generating data---")
        self.dataset.generate(self, verbose=self.verbose)

        if save:
            self.save_data()

    def partition_val_dataset(self, fraction=0.1, bias=0.0):
        if "val_fraction" in self.train_details.__dict__:
            fraction = self.train_details.val_fraction
        if "val_bias" in self.train_details.__dict__:
            bias = self.train_details.val_bias
        self.val_dataset = self.partition_dataset(
            fraction=fraction,
            bias=bias,
            name="val",
        )

    def partition_test_dataset(self, fraction=0.1, bias=0.0):
        if "test_fraction" in self.train_details.__dict__:
            fraction = self.train_details.test_fraction
        if "test_bias" in self.train_details.__dict__:
            bias = self.train_details.test_bias
        self.test_dataset = self.partition_dataset(
            fraction=fraction,
            bias=bias,
            name="test",
        )

    def compute_metrics(self, save=True):
        self.verbose("\n---Computing metrics---")

        if save:
            with h5py.File(join(self.path_to_data, self.fname_metrics), "a") as f:
                if self.mode not in f:
                    group = f.create_group(self.mode)
                else:
                    group = f[self.mode]
                for k, m in self.metrics.items():
                    self.loggers.on_task_update("metrics")
                    m.compute(self, verbose=self.verbose)
                    m.save(group)
        else:
            for k, m in self.metrics.items():
                self.loggers.on_task_update("metrics")
                m.compute(self, verbose=self.verbose)

    def zip(self, to_zip=None):
        to_zip = to_zip or self.__files__
        if "config.pickle" not in to_zip:
            to_zip.append("config.pickle")

        zip = zipfile.ZipFile(
            os.path.join(self.path_to_summary, self.name + ".zip"), mode="w"
        )
        for root, _, files in os.walk(self.path_to_data):
            for f in files:
                if f in to_zip:
                    p = os.path.basename(root)
                    zip.write(os.path.join(root, f), os.path.join(p, f))
        zip.close()

    def save(self, label_with_mode=True):

        self.save_config()
        self.save_data(label_with_mode=label_with_mode)
        self.save_model(label_with_mode=label_with_mode)
        self.save_metrics(label_with_mode=label_with_mode)
        with open(join(self.path_to_data, self.fname_logger), "w") as f:
            self.loggers.save(f)

    def load(self, label_with_mode=True):
        try:
            self.load_config()
        except:
            print("Unable to load config, proceed anyway.")
            pass
        try:
            self.load_data(label_with_mode=label_with_mode)
        except:
            print("Unable to load data, proceed anyway.")
            pass
        try:
            self.load_model(label_with_mode=label_with_mode)
        except:
            print("Unable to load model, proceed anyway.")
            pass
        try:
            self.load_metrics(label_with_mode=label_with_mode)
        except:
            print("Unable to load metrics, proceed anyway.")
            pass
        if exists(join(self.path_to_data, self.fname_logger)):
            if self.loggers is not None:
                with open(join(self.path_to_data, self.fname_logger), "r") as f:
                    self.loggers.load(f)

    # Other methods
    def partition_dataset(self, fraction=0.1, bias=0.0, name="val"):
        self.verbose(f"\n---Partitioning {name}-data---")
        partition = self.dataset.partition(type="random", fraction=fraction, bias=bias)
        if np.sum(partition.network_weights) == 0:
            self.verbose("After partitioning, partition is still empty.")
            partition = None
        real_fraction = 0
        for k in range(partition.networks.size):
            for w in partition.weights[k].data:
                real_fraction += (
                    (w > 0).mean() / partition.weights[k].size / partition.networks.size
                )
        self.verbose(f"Fraction of partitionned samples: {np.round(real_fraction, 4)}")

        return partition

    @classmethod
    def from_file(cls, path_to_config):
        with open(path_to_config, "rb") as config_file:
            config = pickle.load(config_file)
        return cls(config)

    @classmethod
    def unzip(cls, path_to_zip, destination=None, label_with_mode=True):
        zip = zipfile.ZipFile(path_to_zip, mode="r")
        path_to_data, _ = os.path.split(zip.namelist()[0])
        destination = destination or "."
        zip.extractall(path=destination)
        cls = cls.from_file(os.path.join(path_to_data, "config.pickle"))
        cls.path_to_data = path_to_data
        cls.load(label_with_mode=label_with_mode)
        shutil.rmtree(path_to_data)
        return cls

    def clean(self):
        paths = os.listdir(self.path_to_data)

        for p in paths:
            p = os.path.join(self.path_to_data, p)
            os.remove(p)

    def save_data(self, label_with_mode=True):
        with h5py.File(join(self.path_to_data, self.fname_data), "w") as f:
            if label_with_mode:
                for mode in self.all_modes:
                    self._dataset[mode].save(f, name=f"{mode}-train")
                    if mode in self._val_dataset:
                        self._val_dataset[mode].save(f, name=f"{mode}-val")
                    if mode in self._test_dataset:
                        self._test_dataset[mode].save(f, name=f"{mode}-test")
            else:
                self._dataset[self.mode].save(f, name="train")
                if self.mode in self._val_dataset:
                    self._val_dataset[self.mode].save(f, name="val")
                if self.mode in self._test_dataset:
                    self._test_dataset[self.mode].save(f, name="test")

    def save_model(self, label_with_mode=True):

        # saving optimizer
        if label_with_mode:
            fname = self.mode + "_" + self.fname_history
        else:
            fname = self.fname_history
        self.model.nn.save_history(join(self.path_to_data, fname))

        # saving optimizer
        if label_with_mode:
            fname = self.mode + "_" + self.fname_optim
        else:
            fname = self.fname_optim
        self.model.nn.save_optimizer(join(self.path_to_data, fname))

        # saving model
        if label_with_mode:
            fname = self.mode + "_" + self.fname_model
        else:
            fname = self.fname_model
        self.model.nn.save_weights(join(self.path_to_data, fname))

    def save_metrics(self, label_with_mode=True):
        with h5py.File(join(self.path_to_data, self.fname_metrics), "a") as f:
            if label_with_mode:
                if self.mode not in f:
                    group = f.create_group(self.mode)
                else:
                    group = f[self.mode]
            else:
                group = f
            for k, m in self.metrics.items():
                m.save(group, name=k)

    def save_config(self):
        with open(join(self.path_to_data, self.fname_config), "wb") as f:
            pickle.dump(self.config, f)

    def load_data(self, label_with_mode=True):
        if exists(join(self.path_to_data, self.fname_data)):
            with h5py.File(join(self.path_to_data, self.fname_data), "r") as f:
                for k, v in f.items():
                    if label_with_mode:
                        mode, name = k.split("-")
                    else:
                        name = k
                        mode = "main"
                    if name == "train":
                        self._dataset[mode].load(v)
                    elif name == "val":
                        if mode not in self._val_dataset:
                            self._val_dataset[mode] = get_dataset(
                                self._dataset[mode].config
                            )
                        self._val_dataset[mode].load(v)
                    elif name == "test":
                        if mode not in self._test_dataset:
                            self._test_dataset[mode] = get_dataset(
                                self._dataset[mode].config
                            )
                        self._test_dataset[mode].load(v)
                    else:
                        raise ValueError(f"Invalid name `{name}` while loading data.")
        else:
            self.verbose("Loading data: Did not find data to load.")

    def load_model(self, restore_best=True, label_with_mode=True):
        # loading history
        if label_with_mode:
            fname = self.mode + "_" + self.fname_history
        else:
            fname = self.fname_history
        if exists(join(self.path_to_data, fname)):
            self.model.nn.load_history(join(self.path_to_data, fname))
        else:
            self.verbose("Loading model: Did not find history to load.")

        # loading optimizer
        if label_with_mode:
            fname = self.mode + "_" + self.fname_optim
        else:
            fname = self.fname_optim
        if exists(join(self.path_to_data, fname)):
            self.model.nn.load_optimizer(join(self.path_to_data, fname))
        else:
            self.verbose("Loading model: Did not find optimizer to load.")

        # loading model
        if label_with_mode:
            fname = self.mode + "_" + self.fname_model
        else:
            fname = self.fname_model
        if restore_best and exists(self.path_to_best):
            self.model.nn.load_weights(self.path_to_best)
        elif exists(join(self.path_to_data, fname)):
            self.model.nn.load_weights(join(self.path_to_data, fname))
        else:
            self.verbose("Loading model: Did not find model to load.")

    def load_metrics(self, label_with_mode=True):
        if exists(join(self.path_to_data, self.fname_metrics)):
            with h5py.File(join(self.path_to_data, self.fname_metrics), "r") as f:
                if label_with_mode:
                    if self.mode not in f:
                        return
                    else:
                        group = f[self.mode]
                else:
                    group = f
                for k in self.metrics.keys():
                    self.metrics[k].load(group, name=k)
        else:
            self.verbose("Loading metrics: Did not find metrics to load.")

    def load_config(self, best=True):
        if exists(join(self.path_to_data, self.fname_config)):
            with open(join(self.path_to_data, self.fname_config), "rb") as f:
                self.config = pickle.load(f)
        else:
            self.verbose("Loading config: Did not find config to load.")

    # Other attributes
    @property
    def dataset(self):
        return self._dataset[self._mode]

    @dataset.setter
    def dataset(self, dataset):
        self._dataset[self._mode] = dataset

    @property
    def val_dataset(self):
        if self._mode in self._val_dataset:
            return self._val_dataset[self._mode]
        else:
            return None

    @val_dataset.setter
    def val_dataset(self, val_dataset):
        self._val_dataset[self._mode] = val_dataset

    @property
    def test_dataset(self):
        if self._mode in self._test_dataset:
            return self._test_dataset[self._mode]
        else:
            return None

    @test_dataset.setter
    def test_dataset(self, test_dataset):
        self._test_dataset[self._mode] = test_dataset

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == self.mode:
            return
        if mode in self._dataset:
            self.verbose(f"Changing mode `{self.mode}` to `{mode}`.")
            self._mode = mode
        else:
            self.verbose(f"Dataset mode `{mode}` not available, kept `{self.mode}`.")
