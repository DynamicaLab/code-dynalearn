import numpy as np

from .config import Config


class DynamicsConfig(Config):
    @classmethod
    def sis(cls):
        cls = cls()
        cls.name = "SIS"
        cls.infection = 0.04
        cls.recovery = 0.08
        cls.init_param = None
        return cls

    @classmethod
    def sir(cls):
        cls = cls()
        cls.name = "SIR"
        cls.infection = 0.04
        cls.recovery = 0.08
        cls.init_param = None
        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.name = "PlanckSIS"
        cls.temperature = 6.0
        cls.recovery = 0.08
        cls.init_param = None

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.name = "AsymmetricSISSIS"
        cls.infection1 = 0.01
        cls.infection2 = 0.012
        cls.recovery1 = 0.19
        cls.recovery2 = 0.22
        cls.coupling = 50.0
        cls.boost = "source"
        cls.init_param = None

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.name = "DSIR"
        cls.infection_prob = 2.5 / 2.3
        cls.recovery_prob = 1.0 / 7.5
        cls.infection_type = 2
        cls.density = 10000
        epsilon = 1e-5
        cls.init_param = np.array([1 - epsilon, epsilon, 0])
        return cls

    @classmethod
    def incsir(cls):
        cls = cls()
        cls.name = "IncSIR"
        cls.infection_prob = 2.5 / 2.3
        cls.recovery_prob = 1.0 / 7.5
        cls.infection_type = 2
        cls.density = 10000
        epsilon = 1e-5
        cls.init_param = epsilon
        return cls
