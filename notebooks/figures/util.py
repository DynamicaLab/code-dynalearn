import os

from dynalearn.experiments import Experiment


def load_experiments(path_to_summaries, exp_names={}):
    exp = {}
    for k, n in exp_names.items():
        p = os.path.join(path_to_summaries, n + ".zip")
        if not os.path.exists(p):
            print(f"Did not find file `{n}.zip`, kept proceding.")
        else:
            exp[k] = Experiment.unzip(p, label_with_mode=True)
            exp[k].load()
    return exp