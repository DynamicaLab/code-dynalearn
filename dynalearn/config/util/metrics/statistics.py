import numpy as np

from dynalearn.config import Config


class StatisticsConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.max_num_points = 10000
        cls.maxlag = 1
        return cls
