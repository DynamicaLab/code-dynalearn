from dynalearn.config import Config


class ForecastConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.num_steps = [1]
        return cls
