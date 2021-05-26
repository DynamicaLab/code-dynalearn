import numpy as np
from itertools import product


class Config:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __str__(self):
        return self.to_string()

    def __setitem__(self, key, val):
        key = key.split("/")
        if len(key) == 1:
            setattr(self, key[0], val)
        else:
            config = getattr(self, key[0])
            key = "/".join(key[1:])
            config[key] = val

    def __getitem__(self, key):
        key = key.split("/")
        if len(key) == 1:
            return getattr(self, key[0])
        else:
            config = getattr(self, key[0])
            key = "/".join(key[1:])
            return config[key]

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def to_string(self, prefix=""):
        string = ""
        for k, v in self.__dict__.items():
            if issubclass(v.__class__, Config):
                string += prefix + f"{k}:\n"
                string += "{0}\n".format(v.to_string(prefix=prefix + "\t"))
            else:
                string += prefix + f"{k}: {v.__str__()}\n"
        return string

    def get_state_dict(self):
        state_dict = {}
        for k, v in self.__dict__.items():
            if k != "_state_dict":
                if issubclass(v.__class__, Config):
                    v_dict = v.state_dict
                    for kk, vv in v_dict.items():
                        state_dict[k + "/" + kk] = vv
                else:
                    state_dict[k] = v
        return state_dict

    @property
    def state_dict(self):
        return self.get_state_dict()

    def has_list(self):
        for k, v in self.state_dict.items():
            if isinstance(v, list):
                return True
        return False

    def merge(self, config):
        for k, v in config.__dict__.items():
            self.__dict__[k] = v

    def copy(self):
        config_copy = self.__class__()
        for k, v in self.__dict__.items():
            if issubclass(v.__class__, Config) or isinstance(v, (np.ndarray, list)):
                setattr(config_copy, k, v.copy())
            else:
                setattr(config_copy, k, v)
        return config_copy

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()
