import psutil

from numpy import mean, round
from .logger import Logger


class MemoryLogger(Logger):
    def __init__(self, unit="gb"):
        if unit == "b":
            self.factor = 1
        elif unit == "kb":
            self.factor = 1024
        elif unit == "mb":
            self.factor = 1024 ** 2
        elif unit == "gb":
            self.factor = 1024 ** 3
        else:
            raise ValueError(
                f"`{unit}` is an invalid unit, valid units are `[b, kb, mb, gb]`"
            )
        Logger.__init__(self)

    def on_task_end(self):
        self.log["min"] = min(self.all)
        self.log["max"] = max(self.all)
        self.log["mean"] = mean(self.all)

    def on_task_update(self, stepname=None):
        memory_usage = round(psutil.virtual_memory().used / self.factor, 4)
        if f"memory-{stepname}" in self.log:
            self.log[f"memory-{stepname}"].append(memory_usage)
        else:
            self.log[f"memory-{stepname}"] = [memory_usage]

    @property
    def all(self):
        _all = []
        for k, v in self.log.items():
            if k[:6] == "memory":
                _all.extend(v)
        return _all
