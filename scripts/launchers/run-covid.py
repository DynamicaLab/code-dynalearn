import os
import json

from dynalearn.utilities import launch_scan

name = "exp"
specs = json.load(open("./specs.json", "r"))["default"]
config = {
    "name": name,
    "path_to_covid": specs["path_to_data"],
    "epochs": 200,
    "type": ["rnn"],
    "model": [
        "DynamicsGATConv",
        "FullyConnectedGNN",
        "IndependentGNN",
        "KapoorConv",
    ],
    "lag": [5],
    "bias": [0.0, 0.25, 0.5, 0.75, 1.0],
    "val_fraction": 0.1,
}
launch_scan(
    name,
    os.path.join(specs["path_to_data"], "covid"),
    os.path.join(specs["path_to_script"], "covid_script.py"),
    command=specs["command"],
    time="15:00:00",
    memory="8G",
    account=specs["account"],
    modules_to_load=specs["modules_to_load"],
    source_path=specs["source_path"],
    config=config,
    devices=specs["devices"],
    verbose=2,
)
