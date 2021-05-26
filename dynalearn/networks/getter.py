from .random import *
from .weight import *
from .transform import *


__networks__ = {
    "GNPNetworkGenerator": GNPNetworkGenerator,
    "GNMNetworkGenerator": GNMNetworkGenerator,
    "BANetworkGenerator": BANetworkGenerator,
    "ConfigurationNetworkGenerator": ConfigurationNetworkGenerator,
}

__weights__ = {
    "EmptyWeightGenerator": EmptyWeightGenerator,
    "UniformWeightGenerator": UniformWeightGenerator,
    "LogUniformWeightGenerator": LogUniformWeightGenerator,
    "NormalWeightGenerator": NormalWeightGenerator,
    "LogNormalWeightGenerator": LogNormalWeightGenerator,
    "DegreeWeightGenerator": DegreeWeightGenerator,
    "BetweennessWeightGenerator": BetweennessWeightGenerator,
}

__transforms__ = {"SparcifierTransform": SparcifierTransform}


def get(config):
    name = config.name
    weight_gen = None
    transforms = []

    if "weights" in config.__dict__:
        if config.weights.name in __weights__:
            weight_gen = __weights__[config.weights.name](config.weights)
        else:
            raise ValueError(
                f"{config.weights.name} is invalid, possible entries are {list(__weights__.keys())}"
            )

    if "transforms" in config.__dict__:
        assert isinstance(config.transforms.names, list)
        for n in config.transforms.names:
            if n in __transforms__:
                transforms.append(__transforms__[n](config.transforms))
            else:
                raise ValueError(
                    f"{n} is invalid, possible entries are {list(__transforms__.keys())}"
                )

    if name in __networks__:
        return __networks__[name](config, weight_gen, transforms)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__networks__.keys())}"
        )
