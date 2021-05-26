import numpy as np
from scipy.stats import poisson
from scipy.optimize import fsolve


class DiscreteDistribution(object):
    def __init__(self, values):
        super(DiscreteDistribution, self).__init__()
        self.val_dict = {k: v for k, v in zip(*values)}
        self.values = values[0]  # Size K
        self.weights = values[1]  # Size K

    def expect(self, func):
        x = func(self.values)  # Size K x D
        return self.weights @ x

    def mean(self):
        f = lambda k: k
        return self.expect(f)

    def var(self):
        f = lambda k: (k - self.mean()) ^ 2
        return self.expect(f)

    def std(self):
        return np.sqrt(self.var())

    def sample(self, num_samples=1):
        return np.random.choice(self.values, size=num_samples, p=self.weights)


def kronecker_distribution(k):
    k = np.array([k])
    p_k = np.array([1])
    return DiscreteDistribution((k, p_k))


def poisson_distribution(avgk, num_k):
    mid_k = np.ceil(avgk)
    if mid_k < num_k:
        down = 0
        up = 2 * num_k + 1
    else:
        down = mid_k - num_k + 1
        up = mid_k + num_k + 2

    k = np.arange(down, up).astype("int")
    p_k = lambda mu: poisson(mu).pmf(k) / np.sum(poisson(mu).pmf(k))
    f_to_solve = lambda mu: np.sum(k * p_k(mu)) - avgk
    l = fsolve(f_to_solve, x0=avgk)[0]

    return DiscreteDistribution((k, p_k(l)))
