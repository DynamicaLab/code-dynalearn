import torch
from .radam import *


__optimizers__ = {
    "Adam": lambda config: lambda p: torch.optim.Adam(
        p,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    ),
    "RAdam": lambda config: lambda p: RAdam(
        p,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    ),
}


def get(config):
    name = config.name

    if name in __optimizers__:
        return __optimizers__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__optimizers__.keys())}"
        )
