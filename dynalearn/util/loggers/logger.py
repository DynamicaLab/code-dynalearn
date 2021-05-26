import json


class Logger:
    def __init__(self):
        self.log = {}

    def on_task_begin(self):
        return

    def on_task_end(self):
        return

    def on_task_update(self, stepname=None):
        return

    def save(self, f):
        json.dump(self.log, f, indent=4)

    def load(self, f):
        self.log = json.load(f)


class LoggerDict:
    def __init__(self, loggers=None):
        self.loggers = loggers or {}
        assert isinstance(self.loggers, dict)

    def __getitem__(self, key):
        return self.loggers[key]

    def keys(self):
        return self.loggers.keys()

    def values(self):
        return self.loggers.values()

    def items(self):
        return self.loggers.items()

    def on_task_begin(self):
        for l in self.values():
            l.on_task_begin()

    def on_task_end(self):
        for l in self.values():
            l.on_task_end()

    def on_task_update(self, stepname=None):
        for l in self.values():
            l.on_task_update(stepname)

    def save(self, f):
        log_dict = {}
        for k, l in self.items():
            log_dict[k] = l.log
        json.dump(log_dict, f, indent=4)

    def load(self, f):
        log_dict = json.load(f)
        for k, v in log_dict.items():
            for _k, _v in self.items():
                if _k == k:
                    _v.log = v
                    break
