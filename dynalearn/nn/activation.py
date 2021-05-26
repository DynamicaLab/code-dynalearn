import torch.nn as nn

__activations__ = {
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


def get(name):
    if name in __activations__:
        return __activations__[name]
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__activations__.keys())}"
        )
