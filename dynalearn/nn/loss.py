import torch
import numpy as np

from dynalearn.util import onehot


def weighted_cross_entropy(y_true, y_pred, weights=None):
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
    if torch.cuda.is_available():
        y_pred = y_pred.cuda()
        y_true = y_true.cuda()
        weights = weights.cuda()
    weights /= weights.sum()
    y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
    loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
    return loss.sum()


def weighted_dkl(y_true, y_pred, weights=None):
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
    if torch.cuda.is_available():
        y_pred = y_pred.cuda()
        y_true = y_true.cuda()
        weights = weights.cuda()
    weights = weights / torch.sum(weights)
    y_true = torch.clamp(y_true, 1e-15, 1 - 1e-15)
    y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
    loss = weights * torch.sum(y_true * (torch.log(y_true) - torch.log(y_pred)), -1)
    return torch.sum(loss)


def weighted_jsd(y_true, y_pred, weights=None):
    m = 0.5 * (y_true + y_pred)
    return weighted_dkl(y_true, m, weights=weights) + weighted_dkl(
        y_pred, m, weights=weights
    )


def weighted_mse(y_true, y_pred, weights=None):
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
        weights /= weights.sum()
    if torch.cuda.is_available():
        y_pred = y_pred.cuda()
        y_true = y_true.cuda()
        weights = weights.cuda()
    loss = weights * torch.sum((y_true - y_pred) ** 2, axis=-1)
    return loss.sum()


__losses__ = {
    "weighted_cross_entropy": weighted_cross_entropy,
    "weighted_mse": weighted_mse,
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "mse": torch.nn.MSELoss(),
}


def get(loss):
    if loss in __losses__:
        return __losses__[loss]
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__losses__.keys())}"
        )
