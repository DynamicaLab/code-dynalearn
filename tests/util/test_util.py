import unittest
import numpy as np

from dynalearn.utilities import get_dataset_from_timeseries


class UtilitiesTest(unittest.TestCase):
    def test_get_dataset_from_timeseries(self):
        T = 10
        N = 5
        lag = 3
        ts = np.arange(T).reshape(-1, 1, 1).repeat(N, 1)
        x, y = get_dataset_from_timeseries(ts, lag=lag)
        x_ref = np.arange(lag).reshape(-1, 1, 1).repeat(N, 1).T
        self.assertTrue(np.all(x[0] == x_ref))
        self.assertTrue(np.all(y[0] == lag))
