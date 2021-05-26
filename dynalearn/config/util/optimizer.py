from dynalearn.config import Config


class OptimizerConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()

        cls.name = "RAdam"
        cls.lr = 1.0e-3
        cls.weight_decay = 1.0e-4
        cls.betas = (0.9, 0.999)
        cls.eps = 1.0e-8
        cls.amsgrad = False

        return cls
