import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


class KernelDensityEstimator:
    def __init__(self, samples=None, max_num_samples=-1):
        self.max_num_samples = max_num_samples

        self.dim = None
        self.kde = None
        self._mean = None
        self._std = None
        self._norm = None
        self._index = None
        if samples is not None and len(samples) > 0:
            assert isinstance(samples, list)

            self.samples = samples
            if isinstance(samples[0], np.ndarray):
                self.dim = np.prod(samples[0].shape)
            elif isinstance(samples[0], (int, float)):
                self.dim = 1
            for s in samples:
                if isinstance(s, (int, float)):
                    s = np.array([s])
                s = s.reshape(-1)
                assert s.shape[0] == self.dim

            self.get_kde()

    def pdf(self, x):
        if isinstance(x, list):
            if len(x) == 0:
                return np.array([1.0])
            x = np.array(x)
            x = x.reshape(x.shape[0], self.dim).T

        x = x.reshape(self.dim, -1)

        assert x.shape[0] == self.dim or self.dim is None

        if self.kde is None:
            return np.ones(x.shape[-1]) / self._norm
        else:
            y = (x[self._index] - self._mean[self._index]) / self._std[self._index]
            # p = self.kde.pdf(y) / self._norm
            p = np.exp(self.kde.score_samples(y.T)) / self._norm
            assert np.all(np.logical_and(p >= 0, p <= 1)), "Encountered invalid value."
            return p

    def get_kde(self):
        if len(self.samples) <= 1:
            self._norm = 1
            return
        x = np.array(self.samples)
        x = x.reshape(x.shape[0], self.dim).T
        mean = np.expand_dims(x.mean(axis=-1), -1)
        std = np.expand_dims(x.std(axis=-1), -1)
        condition = np.logical_or(std < 1e-8, np.isnan(std))
        if np.all(np.logical_or(std < 1e-8, np.isnan(std))):
            self._norm = len(self.samples)
            return
        self._index = np.where(~condition)[0]
        y = (x[self._index] - mean[self._index]) / std[self._index]
        # self.kde = gaussian_kde(y, bw_method="silverman")
        self.kde = KernelDensity(kernel="gaussian").fit(y.T)
        self._mean = mean
        self._std = std
        # p = self.kde.pdf(y)
        p = np.exp(self.kde.score_samples(y.T))
        self._norm = p.sum()
        assert np.all(p > 0), f"Encountered an invalid value"
        self.samples = []
