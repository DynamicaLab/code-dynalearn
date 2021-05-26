from .callbacks import Callback
from .util import atomic_lambda_save
from dynalearn.util import Verbose


class PeriodicSaveCallback(Callback):
    def __init__(
        self,
        filename,
        *,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        period=1,
        verbose=Verbose(),
        temporary_filename=None,
        atomic_write=True,
        open_mode="wb"
    ):
        super().__init__()
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.temporary_filename = temporary_filename
        self.atomic_write = atomic_write
        self.open_mode = open_mode
        self.best_filename = None

        if self.save_best_only:
            if mode not in ["min", "max"]:
                raise ValueError("Invalid mode '%s'" % mode)
            if mode == "min":
                self.monitor_op = lambda x, y: x < y
                self.current_best = float("Inf")
            elif mode == "max":
                self.monitor_op = lambda x, y: x > y
                self.current_best = -float("Inf")

        self.period = period

    def save_file(self, fd, epoch_number, logs):
        raise NotImplementedError

    def _save_file(self, filename, epoch_number, logs):
        atomic_lambda_save(
            filename,
            self.save_file,
            (epoch_number, logs),
            temporary_filename=self.temporary_filename,
            open_mode=self.open_mode,
            atomic=self.atomic_write,
        )

    def on_epoch_end(self, epoch_number, logs):
        filename = self.filename.format_map(logs)

        if self.save_best_only:
            if self.monitor_op(logs[self.monitor], self.current_best):
                old_best = self.current_best
                self.current_best = logs[self.monitor]
                self.best_filename = filename

                self.verbose(
                    "%s improved from %0.5f to %0.5f, saving file to %s"
                    % (self.monitor, old_best, self.current_best, self.best_filename)
                )
                self._save_file(self.best_filename, epoch_number, logs)
        elif epoch_number % self.period == 0:
            self.verbose("Epoch %d: saving file to %s" % (epoch_number, filename))
            self._save_file(filename, epoch_number, logs)


class PeriodicSaveLambda(PeriodicSaveCallback):
    """
    Call a lambda with a file descriptor after every epoch. See
    :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments'
    descriptions.

    Args:
        func (Callable[[fd, int, dict], None]): The lambda that will be called
        with a file descriptor, the epoch number and the epoch logs.

    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def save_file(self, fd, epoch_number, logs):
        self.func(fd, epoch_number, logs)
