import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from shutil import rmtree
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment


class MetricsTest:
    @property
    def name(self):
        raise NotImplemented()

    def setUp(self):
        self.config = ExperimentConfig.test()
        self.config.networks.num_nodes = 10
        self.config.train_details.num_samples = 15
        self.config.train_details.num_networks = 1
        self.additional_configs()
        self.config.metrics.names = [self.name]
        self.exp = Experiment(self.config, verbose=0)
        self.exp.generate_data()
        self.exp.partition_test_dataset()
        self.exp.partition_val_dataset()

    def tearDown(self):
        rmtree("./test")

    def check_data(selt, data):
        pass

    def additional_configs(self):
        pass

    def test_compute(self):
        self.exp.metrics[self.name].compute(self.exp)
        self.check_data(self.exp.metrics[self.name].data)
