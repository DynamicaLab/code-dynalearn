import h5py
import numpy as np
import argparse
import os
import time

from os.path import exists, join
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment

np.seterr(invalid="ignore")


def check_string_type(s, s_type):
    try:
        s_type(s)
        return True
    except ValueError:
        return False


def get_unkown_args(l):
    d = {}
    k = None
    c_type = list
    for ll in l:
        if ll[:2] == "--":
            if k is not None:
                d[k] = c_type(d[k])
            k = ll[2:]
        else:
            if check_string_type(ll, int):
                ll = int(ll)
            elif check_string_type(ll, float):
                ll = float(ll)
            elif check_string_type(ll, str):
                if ll == "True" or ll == "False":
                    ll = ll == "True"
            else:
                raise ValueError(f"Invalid object `{ll}` with type `{type(ll)}`")
            if k in d:
                d[k].append(ll)
            else:
                d[k] = [ll]
    args = argparse.Namespace()
    for k, v in d.items():
        if len(v) == 1:
            v = v[0]
        setattr(args, k, v)
    return args


def get_metrics(args):
    metrics = args.metrics
    names = []
    for m in metrics:
        if m == "ltp":
            names.extend(["TrueLTPMetrics", "GNNLTPMetrics", "MLELTPMetrics"])
        elif m == "stationary":
            names.extend(["TrueERSSMetrics", "GNNERSSMetrics"])
        elif m == "stats":
            names.extend(["StatisticsMetrics"])
        elif m == "pred":
            names.extend(["PredictionMetrics"])
        elif m == "attention":
            names.extend(["AttentionMetrics"])
        elif m == "attention-nmi":
            names.extend(
                [
                    "AttentionStatesNMIMetrics",
                    "AttentionNodeAttrNMIMetrics",
                    "AttentionEdgeAttrNMIMetrics",
                ]
            )
    return names


def get_dataset(args):
    if args.dynamics == "dsis" or args.dynamics == "dsir":
        if args.weight_type == "plain":
            return dynalearn.config.ContinuousDatasetConfig.plain()
        elif args.weight_type == "structure":
            return dynalearn.config.ContinuousDatasetConfig.structure()
        elif args.weight_type == "state":
            return dynalearn.config.ContinuousDatasetConfig.state()
    else:
        if args.weight_type == "plain":
            return dynalearn.config.DiscreteDatasetConfig.plain()
        elif args.weight_type == "structure":
            return dynalearn.config.DiscreteDatasetConfig.structure()
        elif args.weight_type == "state":
            return dynalearn.config.DiscreteDatasetConfig.state()


parser = argparse.ArgumentParser()

## REQUIRED PARAMETERS
parser.add_argument(
    "--name",
    type=str,
    metavar="EXPNAME",
    help="Name of the experiment.",
    required=True,
)
parser.add_argument(
    "--dynamics",
    type=str,
    metavar="DYNAMICS",
    help="Dynamics of the experiment.",
    choices=[
        "sis",
        "plancksis",
        "sissis",
        "dsir",
    ],
    required=True,
)
parser.add_argument(
    "--network",
    type=str,
    metavar="NETWORK",
    help="Network of the experiment.",
    choices=["gnp", "w_gnp", "ba", "w_ba", "mw_ba", "treeba", "covid"],
    required=True,
)
parser.add_argument(
    "--path_to_data",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
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
    "--tasks",
    type=str,
    metavar="TASKS",
    help="Experiment tasks.",
    nargs="+",
    required=True,
)
parser.add_argument(
    "--metrics",
    type=str,
    metavar="METRICS",
    help="Metrics to compute.",
    nargs="+",
    required=True,
)
parser.add_argument(
    "--to_zip", type=str, metavar="ZIP", help="Data to zip.", nargs="+", required=True
)
parser.add_argument(
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)
parser.add_argument(
    "--seed",
    type=int,
    metavar="VERBOSE",
    help="Verbose.",
    default=int(time.time()),
)


args, unknown = parser.parse_known_args()
others = get_unkown_args(unknown)

config = ExperimentConfig.default(
    args.name,
    args.dynamics,
    args.network,
    path_to_data=args.path_to_data,
    path_to_best=args.path_to_best,
    path_to_summary=args.path_to_summary,
    seed=args.seed,
)
config.metrics.names = tuple(args.metrics)
for k, v in others.__dict__.items():
    if k in config.state_dict:
        config[k] = v

print(config)

# exp = Experiment(config, verbose=args.verbose)
# exp.run()
