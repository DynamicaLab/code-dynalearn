import unittest
import numpy as np

from scipy.stats import gaussian_kde
from dynalearn.datasets.data import KernelDensityEstimator


class KernelDensityEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.num_points = 10
        self.shape = 3
        self.dataset1 = [np.random.randn(self.shape) for i in range(self.num_points)]
        self.dataset2 = [self.dataset1[0] for i in range(self.num_points)]
        self.dataset3 = [np.random.randn(self.shape)]

    def test_init(self):
        kde = KernelDensityEstimator(self.dataset1)
        kde = KernelDensityEstimator(self.dataset2)
        kde = KernelDensityEstimator(self.dataset3)

    def test_mean(self):
        kde = KernelDensityEstimator(self.dataset1)
        mean = np.expand_dims(np.array(self.dataset1).mean(0), axis=-1)
        np.testing.assert_array_almost_equal(kde.mean, mean)

        kde = KernelDensityEstimator(self.dataset2)
        self.assertEqual(kde.mean, None)

        kde = KernelDensityEstimator(self.dataset3)
        self.assertEqual(kde.mean, None)

    def test_std(self):
        kde = KernelDensityEstimator(self.dataset1)
        std = np.expand_dims(np.array(self.dataset1).std(0), axis=-1)
        np.testing.assert_array_almost_equal(kde.std, std)

        kde = KernelDensityEstimator(self.dataset2)
        self.assertEqual(kde.std, None)

        kde = KernelDensityEstimator(self.dataset3)
        self.assertEqual(kde.std, None)

    def test_pdf(self):
        index = np.random.randint(len(self.dataset1))
        x = np.array(self.dataset1).reshape(self.num_points, self.shape).T
        x[index]
        y = (x - np.expand_dims(x.mean(-1), axis=-1)) / np.expand_dims(
            x.std(-1), axis=-1
        )
        my_kde = KernelDensityEstimator(self.dataset1)
        kde = gaussian_kde(y)
        p = kde.pdf(y) / kde.pdf(y).sum()
        np.testing.assert_array_almost_equal(my_kde.pdf(self.dataset1), p)
        for i in range(10):
            p = kde.pdf(x[i])

        my_kde = KernelDensityEstimator(self.dataset2)
        np.testing.assert_array_almost_equal(
            my_kde.pdf(self.dataset2), np.ones(len(self.dataset2)) / len(self.dataset2)
        )

        my_kde = KernelDensityEstimator(self.dataset3)
        np.testing.assert_array_almost_equal(
            my_kde.pdf(self.dataset3), np.ones(len(self.dataset3)) / len(self.dataset3)
        )


if __name__ == "__main__":
    unittest.main()
