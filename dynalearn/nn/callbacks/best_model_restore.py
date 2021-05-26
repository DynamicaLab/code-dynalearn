import warnings

from .callbacks import Callback
from dynalearn.util import Verbose


class BestModelRestore(Callback):
    def __init__(self, *, monitor="val_loss", mode="min", verbose=Verbose()):
        super().__init__()
        self.monitor = monitor

        if mode not in ["min", "max"]:
            raise ValueError("Invalid mode '%s'" % mode)
        if mode == "min":
            self.monitor_op = lambda x, y: x < y
            self.current_best = float("Inf")
        elif mode == "max":
            self.monitor_op = lambda x, y: x > y
            self.current_best = -float("Inf")
        self.best_weights = None
        self.verbose = verbose

    def on_epoch_end(self, epoch_number, logs):
        if self.monitor_op(logs[self.monitor], self.current_best):
            old_best = self.current_best
            self.current_best = logs[self.monitor]

            self.verbose(
                "Epoch %d: %s improved from %0.5f to %0.5f"
                % (epoch_number, self.monitor, old_best, self.current_best)
            )
            self.best_weights = self.get_weight_copies()

    def on_train_end(self, logs):
        if self.best_weights is not None:
            self.verbose("Restoring best model")
            self.model.set_weights(self.best_weights)
        else:
            warnings.warn("No  weights to restore!")

    def get_weight_copies(self):
        weights = self.model.get_weights()
        for k in weights:
            weights[k] = weights[k].cpu().clone()
        return weights
