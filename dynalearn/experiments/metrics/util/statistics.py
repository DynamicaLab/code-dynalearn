import numpy as np


class Statistics:
    def __call__(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x

    @classmethod
    def getter(self, config):
        __all_statistics__ = {
            "Statistics": Statistics,
            "MeanVarStatistics": MeanVarStatistics,
        }

        if config.statistics in __all_statistics__:
            return __all_statistics__[config.statistics]()
        else:
            raise ValueError(
                f"`{config.statistics}` is invalid, valid entries are `{__all_statistics__.key()}`"
            )

    def avg(self, s):
        y = np.mean(s, axis=(0, 1))
        assert y.shape == (s.shape[-1],)
        return y


class MeanVarStatistics(Statistics):
    def __call__(self, x):
        if isinstance(x, list):
            x = np.array(x)
        s = np.zeros((2, x.shape[-1]))
        s[0] = np.mean(x, axis=0)
        s[1] = np.var(x, axis=0)
        return s

    def avg(self, s):
        y = s[0]
        assert y.shape == (s.shape[-1],)
        return y
