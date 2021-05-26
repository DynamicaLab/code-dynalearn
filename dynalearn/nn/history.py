import pickle
import numpy as np


class History:
    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self._epoch_logs = {}
        self._batch_logs = {}

    def reset(self):
        self.epoch = 0
        self.batch = 0
        self._epoch_logs = {}
        self._batch_logs = {}

    def update_epoch(self, logs):
        self._epoch_logs[self.epoch] = logs
        self.epoch += 1

    def update_batch(self, logs):
        self._batch_logs[self.batch] = logs
        self.batch += 1

    def display(self, epoch=None):
        if epoch is None:
            epoch = self.epoch - 1
        log_str = ""
        for k, v in self._epoch_logs[epoch].items():
            if k is not "epoch":
                log_str += f"{k}: {v:.4f}\t"

        return f"\t{log_str}"

    def save(self, file):
        data = {"epoch_logs": self._epoch_logs, "batch_logs": self._batch_logs}
        pickle.dump(data, file, indent=4)

    def load(self, file):
        data = pickle.load(file)
        if "epoch_logs" in dat:
            self._epoch_logs.update(data["epoch_logs"])
        if "batch_logs" in dat:
            self._batch_logs.update(data["batch_logs"])

        self.epoch = max(self._epoch_logs.keys()) + 1
        self.batch = max(self._batch_logs.keys()) + 1

    def get_from_epoch_logs(self, key):
        epochs = list(self._epoch_logs.keys())
        logs = []
        for e, log in self._epoch_logs.items():
            if key in log:
                logs.append(log[key])
            else:
                logs.append(np.nan)

        return epochs, logs

    def get_from_batch_logs(self, key):
        batches = list(self._batch_logs.keys())
        logs = []
        for b, log in self._batch_logs.items():
            if key in log:
                logs.append(log[key])
            else:
                logs.append(np.nan)

        return batches, logs
