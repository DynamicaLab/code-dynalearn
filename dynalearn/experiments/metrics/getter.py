from .metrics import CustomMetrics
from .ltp import *
from .prediction import *
from .statistics import *
from .stationary import *
from .forecast import *
from .attention import *

__metrics__ = {
    "TrueLTPMetrics": TrueLTPMetrics,
    "GNNLTPMetrics": GNNLTPMetrics,
    "MLELTPMetrics": MLELTPMetrics,
    "PredictionMetrics": PredictionMetrics,
    "StatisticsMetrics": StatisticsMetrics,
    "TruePSSMetrics": TruePSSMetrics,
    "GNNPSSMetrics": GNNPSSMetrics,
    "TrueERSSMetrics": TrueERSSMetrics,
    "GNNERSSMetrics": GNNERSSMetrics,
    "TrueForecastMetrics": TrueForecastMetrics,
    "GNNForecastMetrics": GNNForecastMetrics,
    "VARForecastMetrics": VARForecastMetrics,
    "AttentionMetrics": AttentionMetrics,
    "AttentionStatesNMIMetrics": AttentionStatesNMIMetrics,
    "AttentionNodeAttrNMIMetrics": AttentionNodeAttrNMIMetrics,
    "AttentionEdgeAttrNMIMetrics": AttentionEdgeAttrNMIMetrics,
}


def get(config):
    names = config.names
    metrics = {}
    for n in names:
        if n in __metrics__:
            metrics[n] = __metrics__[n](config)
        else:
            metrics[n] = CustomMetrics(config)
    return metrics
