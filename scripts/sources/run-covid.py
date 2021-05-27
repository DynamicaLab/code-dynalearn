import argparse
import dynalearn
import h5py
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import time
import tqdm

from os.path import exists, join


def loading_covid_data(
    experiment, path_to_covid, lag=1, lagstep=1, incidence=True, threshold=False
):
    if incidence:
        dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19cases.h5"), "r")
        num_states = 1
    else:
        dataset = h5py.File(os.path.join(path_to_covid, "spain-covid19.h5"), "r")
        num_states = 3
    X = dataset["weighted-multiplex/data/inputs/d0"][...]
    Y = dataset["weighted-multiplex/data/targets/d0"][...]
    networks = dataset["weighted-multiplex/data/networks/d0"]

    data = {
        "inputs": dynalearn.datasets.DataCollection(name="inputs"),
        "targets": dynalearn.datasets.DataCollection(name="targets"),
        "networks": dynalearn.datasets.DataCollection(name="networks"),
    }
    inputs = np.zeros((X.shape[0] - (lag - 1) * lagstep, X.shape[1], num_states, lag))
    targets = np.zeros((Y.shape[0] - (lag - 1) * lagstep, Y.shape[1], num_states))
    for t in range(inputs.shape[0]):
        x = X[t : t + lag * lagstep : lagstep]
        y = Y[t + lag * lagstep - 1]
        if incidence:
            x = x.reshape(*x.shape, 1)
            y = y.reshape(*y.shape, 1)
        x = np.transpose(x, (1, 2, 0))
        inputs[t] = x
        targets[t] = y
    data["inputs"].add(dynalearn.datasets.StateData(data=inputs))
    data["targets"].add(dynalearn.datasets.StateData(data=targets))
    data["networks"].add(dynalearn.datasets.NetworkData(data=networks))
    pop = data["networks"][0].data.node_attr["population"]
    experiment.dataset.data = data
    experiment.test_dataset = experiment.dataset.partition(
        type="cleancut", ti=335, tf=-1
    )
    experiment.partition_val_dataset()
    return experiment


## REQUIRED PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, metavar="NAME", help="Name of the experiment.", required=True
)
parser.add_argument(
    "--path_to_data",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    required=True,
)
parser.add_argument(
    "--path_to_covid",
    type=str,
    metavar="PATH",
    help="Path to the directory where covid data is saved.",
    required=True,
)
parser.add_argument(
    "--path_to_best",
    type=str,
    metavar="PATH",
    help="Path to the model directory.",
    required=True,
)
parser.add_argument(
    "--path_to_summary",
    type=str,
    metavar="PATH",
    help="Path to the summaries directory.",
    required=True,
)
parser.add_argument(
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)

## OPTIONAL PARAMETERS
parser.add_argument(
    "--epochs",
    type=int,
    metavar="EPOCHS",
    help="Number of epochs to train.",
    default=30,
)
parser.add_argument(
    "--incidence",
    type=int,
    metavar="INCIDENCE",
    help="Using incidence to train the model.",
    default=1,
)
parser.add_argument(
    "--bias",
    type=float,
    metavar="BIAS",
    help="Exponent bias to use.",
    default=0.5,
)
parser.add_argument(
    "--val_fraction",
    type=float,
    metavar="VAL_FRACTION",
    help="Size of the validation dataset relative to the complete dataset.",
    default=0.01,
)
parser.add_argument(
    "--lag",
    type=int,
    metavar="lag",
    help="Size of the windows during training.",
    default=1,
)
parser.add_argument(
    "--lagstep",
    type=int,
    metavar="LAGSTEP",
    help="Step between windows during training.",
    default=1,
)
parser.add_argument(
    "--model",
    type=str,
    metavar="MODEL",
    help="GNN model to use.",
    default="DynamicsGATConv",
)
parser.add_argument(
    "--type",
    type=str,
    metavar="TYPE",
    help="Type of GNN model to use.",
    default="linear",
)
parser.add_argument(
    "--clean",
    type=int,
    metavar="CLEAN",
    help="Clean the experiment before running.",
    default=1,
)
parser.add_argument(
    "--seed", type=int, metavar="SEED", help="Seed of the experiment.", default=-1
)

args = parser.parse_args()

if args.seed == -1:
    args.seed = int(time.time())

config = dynalearn.config.ExperimentConfig.covid(
    args.name,
    args.path_to_data,
    args.path_to_best,
    args.path_to_summary,
    incidence=bool(args.incidence),
)

config.train_details.val_fraction = args.val_fraction
config.train_details.val_bias = args.bias
config.dataset.bias = args.bias
config.model.lag = args.lag
config.model.lagstep = args.lagstep
if args.model != "Kapoor2020":
    config.model.gnn_name = args.model
if args.model == "FullyConnectedGNN":
    config.model.num_nodes = config.networks.num_nodes
elif args.model == "Kapoor2020":
    config.dataset.transforms = dynalearn.config.TransformConfig.kapoor2020()
    config.model = dynalearn.config.TrainableConfig.kapoor()
    config.model.lag = args.lag = 7
    config.dynamics.is_weighted = config.dynamics.is_multiplex = False
    config.model.is_weighted = config.model.is_multiplex = False
    config.model.optimizer.lr = 1e-5
    config.model.optimizer.weight_decay = 5e-4
config.model.type = args.type
config.train_details.epochs = args.epochs
config.train_metrics = []
config.metrics.names = [
    "AttentionMetrics",
    "AttentionStatesNMIMetrics",
    "AttentionNodeAttrNMIMetrics",
    "AttentionEdgeAttrNMIMetrics",
    "GNNForecastMetrics",
    "VARForecastMetrics",
]
config.metrics.num_steps = [1, 7, 14]

# Defining the experiment
experiment = dynalearn.experiments.Experiment(config, verbose=args.verbose)

experiment.clean()
experiment.begin()
experiment.dataset.setup(experiment)
experiment.save_config()

# Training on covid data
experiment.mode = "main"
loading_covid_data(
    experiment,
    args.path_to_covid,
    lag=args.lag,
    lagstep=args.lagstep,
    incidence=bool(args.incidence),
)
experiment.model.nn.history.reset()
experiment.callbacks[0].current_best = np.inf
experiment.train_model(save=False, restore_best=True)
experiment.compute_metrics()
experiment.save(label_with_mode=False)
experiment.end()
experiment.zip(
    to_zip=(
        "config.pickle",
        "data.h5",
        "metrics.h5",
        "history.pickle",
        "model.pt",
        "optim.pt",
    )
)
