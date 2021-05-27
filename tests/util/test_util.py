import unittest
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from dynalearn.util import *


class UtilitiesTest(unittest.TestCase):
    def test_from_binary(self):
        self.assertEqual(from_binary([1, 0, 1]), 5)

    def test_to_binary(self):
        self.assertTrue(np.all(np.array([1, 0, 1]) == to_binary(5)))

    def test_logbase(self):
        self.assertEqual(logbase(4, 2), 2)
        self.assertEqual(logbase(9, 3), 2)
        self.assertEqual(logbase(27, 3), 3)

    def test_from_nary(self):
        self.assertEqual(from_nary([1, 0, 2], base=3), 11)
        self.assertEqual(from_nary([1, 0, 1, 2], base=4), 70)

    def test_to_nary(self):
        self.assertTrue(np.all(to_nary(11, base=3) == np.array([[1, 0, 2]]).T))
        self.assertTrue(np.all(to_nary(70, base=4) == np.array([[1, 0, 1, 2]]).T))

    def test_onehot_numpy(self):
        x = np.array([1, 0, 2])
        x_onehot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertTrue(np.all(x_onehot == onehot_numpy(x)))

    def test_onehot_torch(self):
        x = torch.Tensor([1, 0, 2])
        x_onehot = torch.Tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertTrue(torch.all(x_onehot == onehot_torch(x).cpu()))

    def test_get_dataset_from_timeseries(self):
        T = 10
        N = 5
        lag = 3
        ts = np.arange(T).reshape(-1, 1, 1).repeat(N, 1)
        x, y = get_dataset_from_timeseries(ts, lag=lag)
        x_ref = np.arange(lag).reshape(-1, 1, 1).repeat(N, 1).T
        self.assertTrue(np.all(x[0] == x_ref))
        self.assertTrue(np.all(y[0] == lag))
