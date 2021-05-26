from dynalearn.config import Config


class LTPConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.max_num_sample = 1000
        cls.max_num_points = -1
        return cls
