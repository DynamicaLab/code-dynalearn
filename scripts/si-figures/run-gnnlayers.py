import json
import os
import numpy as np
import sys

sys.path.append("../sources")
from script import launch_scan

sys.path.append("../si-figures")

specs = json.load(open("../sources/specs.json", "r"))["default"]


def launching(config):
    launch_scan(
        name,
        os.path.join(specs["path_to_data"], "gnn-layers"),
        "..sources/run.py",
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
    "dynamics": ["sis", "plancksis", "sissis"],
    "network": ["gnp", "ba"],
    "tasks": (
        "generate_data",
        "partition_val_dataset",
        "train_model",
        "compute_metrics",
    ),
    "metrics": ("ltp"),
    "to_zip": ("config.pickle", "metrics.h5"),
    "train_details/num_samples": 10000,
    "train_details/use_groundtruth": 0,
    "train_details/resampling": 2,
    "train_details/val_bias": 0.5,
    "train_details/val_fraction": 0.01,
    "train_details/train_bias": 0.5,
    "train_details/epochs": 60,
    "networks/num_nodes": 1000,
    "model/gnn_name": [
        "GATConv",
        "SAGEConv",
        "GCNConv",
        "MeanGraphConv",
        "MaxGraphConv",
        "AddGraphConv",
        "KapoorConv",
        "DynamicsGATConv",
    ],
    "weight_type": "state",
    "seed": 0,
}
launching(config)
