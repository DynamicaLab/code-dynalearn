from dynalearn.config import Config


class WeightConfig(Config):
    @classmethod
    def uniform(cls):
        cls = cls()
        cls.name = "UniformWeightGenerator"
        cls.low = 0
        cls.high = 100
        return cls

    @classmethod
    def loguniform(cls):
        cls = cls()
        cls.name = "LogUniformWeightGenerator"
        cls.low = 1e-5
        cls.high = 100
        return cls

    @classmethod
    def normal(cls):
        cls = cls()
        cls.name = "NormalWeightGenerator"
        cls.mean = 100
        cls.std = 5
        return cls

    @classmethod
    def lognormal(cls):
        cls = cls()
        cls.name = "LogNormalWeightGenerator"
        cls.mean = 100
        cls.std = 5
        return cls

    @classmethod
    def degree(cls):
        cls = cls()
        cls.name = "DegreeWeightGenerator"
        cls.mean = 100
        cls.std = 5
        cls.normalized = True
        return cls

    @classmethod
    def betweenness(cls):
        cls = cls()
        cls.name = "BetweennessWeightGenerator"
        cls.mean = 100
        cls.std = 5
        cls.normalized = True
        return cls
