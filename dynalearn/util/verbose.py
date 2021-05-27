import tqdm
from datetime import datetime


class Verbose:
    def __init__(self, filename=None, vtype=0, pbar=None):
        self.filename = filename
        self.to_file = filename is not None
        self.vtype = vtype
        if pbar == "notebook":
            self.pbar = tqdm.tqdm_notebook
        elif pbar is not None:
            self.pbar = pbar
        else:
            self.pbar = tqdm.tqdm
        if self.to_file:
            _file = open(self.filename, "w")
            _file.close()

    def __call__(self, msg):
        self.save_msg(msg)
        self.print_msg(msg)

    def save_msg(self, msg):
        if self.filename is None:
            return
        _file = open(self.filename, "a")
        _file.write(f"{msg}\n")
        _file.close()

    def print_msg(self, msg):
        if self.vtype != 0:
            print(msg)

    def progress_bar(self, name, num_update):
        if self.vtype == 1:
            self.save_msg(name)
            return self.pbar(range(num_update), name)
        else:
            self(name)
            return None
