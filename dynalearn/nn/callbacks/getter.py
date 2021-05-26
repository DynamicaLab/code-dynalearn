from .callbacks import Callback, CallbackList
from .checkpoint import *
from .lr_scheduler import *
from . import lr_scheduler
from .lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler


import inspect

__callbacks__ = {
    "Callback": lambda config: Callback(),
    "CallbackList": lambda config: CallbackList(),
    "BestModelRestore": lambda config: BestModelRestore(),
    "ModelCheckpoint": lambda config: ModelCheckpoint(
        config.path_to_best, save_best_only=True
    ),
    "OptimizerCheckpoint": lambda config: OptimizerCheckpoint(
        config.path_to_best, save_best_only=True
    ),
    "LambdaLR": lambda config: LambdaLR(config.lr_lambda),
    "MultiplicativeLR": lambda config: MultiplicativeLR(config.lr_lambda),
    "StepLR": lambda config: StepLR(config.step_size, gamma=config.gamma),
    "MultiStepLR": lambda config: MultiStepLR(config.milestones, gamma=config.gamma),
    "ExponentialLR": lambda config: ExponentialLR(config.gamma),
    "CosineAnnealingLR": lambda config: CosineAnnealingLR(
        confg.t_max, eta_min=config.eta_min
    ),
}


def get(config):
    names = config.names
    callbacks = []
    for n in names:
        if n in __callbacks__:
            callbacks.append(__callbacks__[n](config))
        else:
            raise ValueError(
                f"{name} is invalid, possible entries are {list(__callbacks__.keys())}"
            )
    return callbacks
