import unittest
import numpy as np

from .templates import *
from dynalearn.dynamics import VARDynamics
from statsmodels.tsa.vector_ar.var_model import VAR


class VARDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.num_states = 1
        self.num_nodes = 5
        self.T = 1000
        self.coefs = np.random.rand(self.num_nodes)
        self.coefs = np.column_stack((self.coefs, 1 - self.coefs))
        self.eps = 0.5
        self.lag = 3
        self.A = 10

        dataset1 = self.get_data(with_y=False)
        dataset2 = self.get_data(with_y=True)

        self.model1 = VARDynamics(self.num_states, lag=self.lag)
        self.model2 = VARDynamics(self.num_states, lag=self.lag)
        self.model1.fit(dataset1)
        self.model2.fit(*dataset2)
        self.ref = VAR(dataset1).fit(maxlags=self.lag)

    def get_data(self, with_y=False):
        X = np.linspace(0, 8 * np.pi, self.T)
        dataset = (
            self.A
            * np.array(
                [
                    c0 * np.sin(X)
                    + c1 * np.exp(-0.1 * X)
                    + self.eps * np.random.randn(self.T)
                    for c0, c1 in self.coefs
                ]
            ).T
        )
        if with_y:
            x = np.array(
                [dataset[t - self.lag : t][::-1] for t in range(self.lag, len(X))]
            )
            self.T = x.shape[0]
            y = dataset[self.lag :].reshape(self.T, -1)
            x = np.transpose(x, (0, 2, 1)).reshape(
                self.T, self.num_nodes, self.num_states, self.lag
            )
            y = y.reshape(self.T, self.num_nodes, self.num_states)
            return (x, y)
        else:
            self.T = dataset.shape[0]
            return dataset.reshape((self.T, self.num_nodes, self.num_states))

    def test_initialstate(self):
        x0 = self.model1.initial_state()
        self.assertEqual(x0.shape, (self.num_nodes, self.num_states, self.lag))

        x0 = self.model2.initial_state()
        self.assertEqual(x0.shape, (self.num_nodes, self.num_states, self.lag))

    def test_predict(self):
        x0 = self.model1.initial_state()
        yp = self.model1.predict(x0)
        self.assertEqual(yp.shape, (self.num_nodes, self.num_states))
        x0 = np.transpose(x0, (1, 2, 0))
        yt = self.ref.forecast(x0.reshape(self.lag, -1), steps=1).reshape(
            self.num_nodes, self.num_states
        )
        np.testing.assert_array_almost_equal(yt, yp, decimal=4)

        x0 = self.model2.initial_state()
        yp = self.model2.predict(x0)
        self.assertEqual(yp.shape, (self.num_nodes, self.num_states))

    def test_sample(self):
        x0 = self.model1.initial_state()
        y = self.model1.sample(x0)
        self.assertEqual(y.shape, (self.num_nodes, self.num_states))

    def test_loglikelihood(self):
        ts = self.model1.data
        logp = self.model1.loglikelihood(ts)
        self.assertTrue(isinstance(logp, float))
        self.assertTrue(logp <= 0)
