from dynalearn.config import Config


class CallbackConfig(Config):
    @classmethod
    def default(cls, path_to_best="./"):
        cls = cls()
        cls.names = ["ModelCheckpoint", "StepLR"]
        cls.step_size = 20
        cls.gamma = 0.5
        cls.path_to_best = path_to_best
        return cls

    @classmethod
    def empty(cls):
        cls = cls()
        cls.names = []
        return cls
