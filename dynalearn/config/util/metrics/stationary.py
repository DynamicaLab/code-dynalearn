import numpy as np

from dynalearn.config import Config


class StationaryConfig(Config):
    @classmethod
    def stocont(cls):
        cls = cls()
        cls.adaptive = True
        cls.num_nodes = 1000
        cls.init_param = {}

        cls.sampler = "SteadyStateSampler"
        cls.burn = 1
        cls.T = 1000
        cls.tol = 500
        cls.num_samples = 5

        cls.statistics = "MeanVarStatistics"
        return cls

    @classmethod
    def metapop(cls):
        cls = cls()

        cls.adaptive = False
        cls.num_nodes = 1000
        cls.init_param = {}

        cls.sampler = "FixedPointSampler"
        cls.initial_burn = 0
        cls.init_epsilon = 1e-5
        cls.mid_burn = 10
        cls.tol = 1e-5
        cls.maxiter = 1000
        cls.num_samples = 20

        cls.statistics = "MeanVarStatistics"
        return cls

    @classmethod
    def test(cls):
        cls = cls.stocont()
        epsilon = 1e-3
        cls.T = 100
        cls.tol = 5
        cls.init_epsilon = 1e-3
        cls.init_param = {
            "absorbing": np.array([1 - epsilon, epsilon]),
            "epidemic": np.array([0, 1]),
        }
        param = np.linspace(0.1, 10.0, 10)
        cls.parameters = {"absorbing": param, "epidemic": param[::-1]}
        cls.num_samples = 1
        return cls

    @classmethod
    def sis(cls):
        cls = cls.stocont()
        epsilon = 1e-3
        cls.init_epsilon = 1e-3
        cls.init_param = {
            "absorbing": np.array([1 - epsilon, epsilon]),
        }
        # param = np.concatenate(
        #     (
        #         np.linspace(0.1, 1.4, 5),
        #         np.linspace(1.5, 3.5, 50),
        #         np.linspace(3.6, 7, 15),
        #     )
        # )
        param = np.linspace(0.1, 7, 50)
        # cls.parameters = {"absorbing": param, "epidemic": param[::-1]}
        cls.parameters = {"absorbing": param}
        return cls

    @classmethod
    def plancksis(cls):
        cls = cls.stocont()
        epsilon = 1e-3
        cls.init_epsilon = 1e-3
        cls.init_param = {
            "absorbing": np.array([1 - epsilon, epsilon]),
            "epidemic": np.array([0, 1]),
        }
        # param = np.concatenate(
        #     (
        #         np.linspace(0.1, 3.0, 10),
        #         np.linspace(3.1, 4.5, 50),
        # np.linspace(4.6, 6, 10),
        #     )
        # )
        param = np.linspace(2.1, 6, 50)

        cls.parameters = {"absorbing": param, "epidemic": param[::-1]}

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls = cls.stocont()
        epsilon = 1e-3
        cls.init_epsilon = 1 - (1 - epsilon) ** 0.5
        cls.init_param = {
            "epidemic": np.array([0, 0, 0, 1]),
            "absorbing": np.array([1 - epsilon, 0, 0, epsilon]),
        }
        # param = np.concatenate(
        #     (
        #         np.linspace(0.1, 1.0, 5),
        #         np.linspace(1.0, 4.5, 50),
        #         np.linspace(4.6, 7, 10),
        #     )
        # )
        param = np.linspace(3.1, 7, 50)

        cls.parameters = {"absorbing": param, "epidemic": param[::-1]}
        return cls

    @classmethod
    def dsir(cls):
        cls = cls.metapop()
        epsilon = 1e-6
        cls.init_param = {
            "absorbing": np.array([1 - epsilon, epsilon, 0]),
        }

        cls.parameters = {"absorbing": np.linspace(0.01, 5, 60)}
        return cls
