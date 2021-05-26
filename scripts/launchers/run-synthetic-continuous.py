import json
import os
import numpy as np
import sys

sys.path.append("../sources")

from script import launch_scan

sys.path.append("../launchers")


specs = json.load(open("../sources/specs.json", "r"))["default"]


def launching(config):
    launch_scan(
        name,
        os.path.join(specs["path_to_data"], "case-study"),
        os.path.join(specs["path_to_script"], "run.py"),
        command=specs["command"],
        time="12:00:00",
        memory="8G",
        account=specs["account"],
        modules_to_load=specs["modules_to_load"],
        source_path=specs["source_path"],
        config=config,
        device=specs["device"],
        verbose=2,
    )


name = "exp"
config = {
    "dynamics": ["dsir"],
    "tasks": (
        "generate_data",
        "partition_val_dataset",
        "train_model",
        "compute_metrics",
    ),
    "to_zip": ("config.pickle", "metrics.h5", "history.pickle", "model.pt", "optim.pt"),
    "num_samples": 100,
    "num_networks": 50,
    "num_nodes": 1000,
    "use_groundtruth": 1,
    "resampling": 100,
    "val_bias": 0.5,
    "val_fraction": 0.01,
    "train_bias": 0.5,
    "epochs": 60,
    "weight_type": "state",
}

config["network"] = ["w_ba"]
config["metrics"] = ("pred", "stationary")
launching(config)

# config["network"] = ["w_gnp"]
# config["metrics"] = "pred"
# launching(config)
