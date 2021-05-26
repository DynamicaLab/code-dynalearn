from dynalearn.datasets import (
    DiscreteDataset,
    DiscreteStructureWeightDataset,
    DiscreteStateWeightDataset,
    ContinuousDataset,
    ContinuousStructureWeightDataset,
    ContinuousStateWeightDataset,
)


__datasets__ = {
    "DiscreteDataset": DiscreteDataset,
    "DiscreteStructureWeightDataset": DiscreteStructureWeightDataset,
    "DiscreteStateWeightDataset": DiscreteStateWeightDataset,
    "ContinuousDataset": ContinuousDataset,
    "ContinuousStructureWeightDataset": ContinuousStructureWeightDataset,
    "ContinuousStateWeightDataset": ContinuousStateWeightDataset,
}


def get(config):
    name = config.name
    if name in __datasets__:
        return __datasets__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__datasets__.keys())}"
        )
