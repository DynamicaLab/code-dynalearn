from dynalearn.config import Config


class PredictionConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.max_num_points = 1e4
        return cls
