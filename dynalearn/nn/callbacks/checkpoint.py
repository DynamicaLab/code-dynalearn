import warnings

from .periodic import PeriodicSaveCallback
from .lr_scheduler import _PyTorchLRSchedulerWrapper, ReduceLROnPlateau


class ModelCheckpoint(PeriodicSaveCallback):
    """
    Save the model after every epoch. See
    :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.
    Args:
        restore_best (bool): If `restore_best` is true, the weights of the network will be reset to
            the last best checkpoint done. This option only works when `save_best_only` is also true.
            (Default value = False)
    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def __init__(self, *args, restore_best=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.restore_best = restore_best
        if self.restore_best and not self.save_best_only:
            raise ValueError(
                "The 'restore_best' argument only works when 'save_best_only' is also true."
            )

    def save_file(self, fd, epoch_number, logs):
        self.model.save_weights(fd)

    def on_train_end(self, logs):
        if self.restore_best:
            if self.best_filename is not None:
                self.verbose("Restoring model from %s" % self.best_filename)
                self.model.load_weights(self.best_filename)
            else:
                warnings.warn("No  weights to restore!")


class OptimizerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of the optimizer after every epoch. The optimizer can be reloaded as follows.
    .. code-block:: python
        model = Model(model, optimizer, loss_function)
        model.load_optimizer_state(filename)
    See :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.
    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def save_file(self, fd, epoch_number, logs):
        self.model.save_optimizer_state(fd)
