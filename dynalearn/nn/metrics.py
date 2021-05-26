import torch

from sklearn.metrics import r2_score
from dynalearn.util import onehot
from .loss import weighted_cross_entropy

EPSILON = 1e-8


def model_entropy(y_true, y_pred, weights=None):
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    if weights is None:
        x = -torch.mean((y_pred * torch.log(y_pred)).sum(-1))
    else:
        weights /= weights.sum()
        x = -torch.sum(weights * (y_pred * torch.log(y_pred)).sum(-1))
    return x


def relative_entropy(y_true, y_pred, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_pred, y_true.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    cross_entropy = weighted_cross_entropy(y_true, y_pred, weights=weights)
    entropy = weighted_cross_entropy(y_true, y_true, weights=weights)
    return cross_entropy - entropy


def approx_relative_entropy(y_true, y_pred, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_pred, y_true.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    cross_entropy = weighted_cross_entropy(y_true, y_pred, weights=weights)
    entropy = weighted_cross_entropy(y_pred, y_pred, weights=weights)
    return entropy - cross_entropy


def jensenshannon(y_true, y_pred, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_true, y_pred.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    m = 0.5 * (y_true + y_pred)
    return 0.5 * (relative_entropy(y_true, m) + relative_entropy(y_pred, m))


def acc(y_true, y_pred, weights=None):
    x = y_true.cpu().detach().numpy()
    y = y_pred.cpu().detach().numpy()
    a = r2_score(x, y)
    return torch.tensor(a, dtype=torch.float)


__metrics__ = {
    "model_entropy": model_entropy,
    "relative_entropy": relative_entropy,
    "approx_relative_entropy": approx_relative_entropy,
    "jensenshannon": jensenshannon,
    "acc": acc,
}


def get(names):
    metrics = {}
    for n in names:
        if n in __metrics__:
            metrics[n] = __metrics__[n]
        else:
            raise ValueError(
                f"{name} is invalid, possible entries are {list(__metrics__.keys())}"
            )
    return metrics
