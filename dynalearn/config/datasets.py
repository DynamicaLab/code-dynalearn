from .config import Config


class DiscreteDatasetConfig(Config):
    @classmethod
    def get_config(cls, weight_type="state", **kwargs):
        return getattr(cls, weight_type)(**kwargs)

    @classmethod
    def plain(cls, **kwargs):
        cls = cls()
        cls.name = "DiscreteDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def structure(cls, use_strength=True, **kwargs):
        cls = cls()
        cls.name = "DiscreteStructureWeightDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        return cls

    @classmethod
    def state(cls, use_strength=True, compounded=True, **kwargs):
        cls = cls()
        cls.name = "DiscreteStateWeightDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        cls.compounded = compounded
        return cls


class ContinuousDatasetConfig(Config):
    @classmethod
    def get_config(cls, weight_type="state", **kwargs):
        return getattr(cls, weight_type)(**kwargs)

    @classmethod
    def plain(cls, **kwargs):
        cls = cls()
        cls.name = "ContinuousDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def structure(cls, use_strength=True, **kwargs):
        cls = cls()
        cls.name = "ContinuousStructureWeightDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        return cls

    @classmethod
    def state(
        cls, use_strength=True, compounded=False, reduce=False, total=True, **kwargs
    ):
        cls = cls()
        cls.name = "ContinuousStateWeightDataset"
        cls.modes = ["main"]
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        cls.compounded = compounded
        cls.total = total
        cls.reduce = reduce
        cls.max_num_points = -1
        return cls


class TransformConfig(Config):
    @classmethod
    def kapoor2020(cls):
        cls = cls()
        cls.names = ["ThresholdNetworkTransform"]
        cls.threshold = 32
        cls.collapse = True
        return cls
