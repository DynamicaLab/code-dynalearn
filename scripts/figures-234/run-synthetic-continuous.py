import json
import os
import numpy as np
import sys

sys.path.append("../sources")

from script import launch_scan

sys.path.append("../figures-234")


specs = json.load(open("../sources/specs.json", "r"))["default"]


def launching(config):
    launch_scan(
        name,
        os.path.join(specs["path_to_data"], "case-study"),
        "../sources/run.py",
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
    "train_details/num_samples": 100,
    "train_details/num_networks": 50,
    "train_details/use_groundtruth": 1,
    "train_details/resampling": 100,
    "train_details/val_bias": 0.5,
    "train_details/val_fraction": 0.01,
    "train_details/train_bias": 0.5,
    "train_details/epochs": 60,
    "networks/num_nodes": 1000,
    "weight_type": "state",
}

config["network"] = ["w_ba"]
config["metrics"] = ("pred", "stationary")
launching(config)

config["network"] = ["w_gnp"]
config["metrics"] = "pred"
launching(config)
