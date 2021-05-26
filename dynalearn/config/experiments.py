import dynalearn as dl
import numpy as np
import time
import os

from dynalearn.config import *
from .util import TrainingConfig, CallbackConfig

network_config = {
    "gnp": NetworkConfig.gnp(),
    "ba": NetworkConfig.ba(),
    "w_gnp": NetworkConfig.w_gnp(),
    "w_ba": NetworkConfig.w_ba(),
    "mw_ba": NetworkConfig.mw_ba(),
}
dynamics_config = {
    "sis": DynamicsConfig.sis(),
    "plancksis": DynamicsConfig.plancksis(),
    "sissis": DynamicsConfig.sissis(),
    "dsir": DynamicsConfig.dsir(),
}
gnn_config = {
    "sis": TrainableConfig.sis(),
    "plancksis": TrainableConfig.plancksis(),
    "sissis": TrainableConfig.sissis(),
    "dsir": TrainableConfig.dsir(),
}
metrics_config = {
    "sis": MetricsConfig.sis(),
    "plancksis": MetricsConfig.plancksis(),
    "sissis": MetricsConfig.sissis(),
    "dsir": MetricsConfig.dsir(),
}
trainingmetrics = {
    "sis": ["jensenshannon", "model_entropy"],
    "plancksis": ["jensenshannon", "model_entropy"],
    "sissis": ["jensenshannon", "model_entropy"],
    "dsir": ["jensenshannon", "model_entropy"],
}


class ExperimentConfig(Config):
    @classmethod
    def default(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls()
        if dynamics not in dynamics_config:
            raise ValueError(
                f"{dynamics} is invalid, valid entries are {list(dynamics_config.keys())}"
            )
        if network not in network_config:
            raise ValueError(
                f"{network} is invalid, valid entries are {list(network_config.keys())}"
            )
        cls.name = name

        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)

        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)

        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)
        cls.dynamics = dynamics_config[dynamics]
        cls.networks = network_config[network]
        cls.model = gnn_config[dynamics]
        if cls.networks.is_weighted:
            cls.dynamics.is_weighted = True
            cls.model.is_weighted = True
        else:
            cls.dynamics.is_weighted = False
            cls.model.is_weighted = False
        if cls.networks.is_multiplex:
            cls.dynamics.is_multiplex = True
            cls.model.is_multiplex = True
            cls.model.network_layers = cls.networks.layers
        else:
            cls.dynamics.is_multiplex = False
            cls.model.is_multiplex = False

        cls.metrics = metrics_config[dynamics]
        cls.train_metrics = trainingmetrics[dynamics]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def stocont(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls.default(
            name,
            dynamics,
            network,
            path_to_data=path_to_data,
            path_to_best=path_to_best,
            path_to_summary=path_to_summary,
            seed=seed,
        )
        cls.dataset = DiscreteDatasetConfig.state()
        cls.train_details = TrainingConfig.discrete()

        return cls

    @classmethod
    def metapop(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls.default(
            name,
            dynamics,
            network,
            path_to_data=path_to_data,
            path_to_best=path_to_best,
            path_to_summary=path_to_summary,
            seed=seed,
        )
        cls.dataset = ContinuousDatasetConfig.state(
            compounded=False, reduce=False, total=True
        )
        cls.train_details = TrainingConfig.continuous()

        return cls

    @classmethod
    def covid(
        cls,
        name,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        incidence=True,
        seed=None,
    ):
        cls = cls()
        cls.name = name

        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)

        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)

        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)

        cls.networks = NetworkConfig.mw_ba(num_nodes=52)
        if incidence:
            cls.dynamics = DynamicsConfig.incsir()
            cls.model = TrainableConfig.incsir()
        else:
            cls.dynamics = DynamicsConfig.dsir()
            cls.model = TrainableConfig.dsir()
        if cls.networks.is_weighted:
            cls.dynamics.is_weighted = True
            cls.model.is_weighted = True
        else:
            cls.dynamics.is_weighted = False
            cls.model.is_weighted = False
        if cls.networks.is_multiplex:
            cls.dynamics.is_multiplex = True
            cls.model.is_multiplex = True
            cls.model.network_layers = cls.networks.layers
        else:
            cls.dynamics.is_multiplex = False
            cls.model.is_multiplex = False

        cls.metrics = MetricsConfig.covid()
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.dataset = ContinuousDatasetConfig.state(
            compounded=False, reduce=False, total=True, use_strength=True
        )
        cls.train_details = TrainingConfig.continuous()

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def test(
        cls,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
    ):
        cls = cls()
        cls.name = "test"
        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)
        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)
        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)
        cls.dataset = DiscreteDatasetConfig.state()
        cls.networks = NetworkConfig.gnp(1000, 4.0 / 999.0)
        cls.dynamics = DynamicsConfig.sis()
        cls.model = TrainableConfig.sis()
        cls.train_details = TrainingConfig.test()
        cls.metrics = MetricsConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
