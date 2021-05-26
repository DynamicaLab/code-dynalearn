import numpy as np
from scipy.special import lambertw


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def constant(l, p):
    return np.ones(l.shape) * p


def independent(l, p):
    return 1 - (1 - p) ** l


def threshold(l, k, beta, mu):
    k = np.sum(np.array(neighbor_state), axis=0)
    l = neighbor_state[1]
    p = sigmoid(beta * (l / k - mu))
    p[k == 0] = 0
    return p


def nonlinear(l, tau, alpha):
    p = (1 - (1 - tau) ** l) ** alpha
    return p


def sine(l, tau, epsilon, period):
    p = (1 - (1 - tau) ** l) * (1 - epsilon * (np.sin(np.pi * l / period)) ** 2)
    return p


def planck(l, temperature):
    gamma = (lambertw(-3 * np.exp(-3)) + 3).real
    Z = gamma ** 3 * temperature ** 3 / (np.exp(gamma) - 1)
    p = np.zeros(l.shape)
    p[l > 0] = l[l > 0] ** 3 / (np.exp(l[l > 0] / temperature) - 1) / Z
    p[l == 0] = 0
    return p
