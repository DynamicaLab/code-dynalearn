import networkx as nx
import numpy as np
import torch

from numba import jit
from cmath import exp, log
from math import log as mlog


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def from_binary(x):
    x = np.array(x)
    n = np.arange(x.shape[0])[::-1]
    return (x * 2 ** (n)).sum()


def to_binary(x, max_val=None):
    max_val = max_val or 2 ** np.floor(np.log2(x) + 1)
    r = np.zeros(np.log2(max_val).astype("int"))
    r0 = x
    while r0 > 0:
        y = np.floor(np.log2(r0)).astype("int")
        r[y] = 1
        r0 -= 2 ** y
    return r[::-1]


def logbase(x, base=np.e):
    return np.log(x) / np.log(base)


def from_nary(x, axis=0, base=2):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    else:
        x = np.array(x)
    n = np.arange(x.shape[axis])[::-1]
    n = n.reshape(*[s if i == axis else 1 for i, s in enumerate(x.shape)])
    return (x * base ** (n)).sum(axis)


def to_nary(x, base=2, dim=None):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    if dim is None:
        max_val = base ** np.floor(logbase(np.max(x), base) + 1)
        dim = int(logbase(max_val, base))
    y = np.zeros([dim, *x.shape])
    for idx, xx in np.ndenumerate(x):
        r = np.zeros(dim)
        r0 = xx
        while r0 > 0:
            b = int(np.floor(logbase(r0, base)))
            r[b] += 1
            r0 -= base ** b
        y.T[idx] = r[::-1]
    return y


def all_combinations(n, k):
    t = n
    h = 0
    a = [0] * k
    a[0] = n
    res = []
    res.append(a.copy())
    while a[k - 1] != n:
        if t != 1:
            h = 0
        t = a[h]
        a[h] = 0
        a[0] = t - 1
        a[h + 1] += 1
        h += 1
        res.append(a.copy())
    return res


@jit(nopython=True)
def numba_all_combinations(n, k):
    t = n
    h = 0
    a = [0] * k
    a[0] = n
    res = []
    res.append(a.copy())
    while a[k - 1] != n:
        if t != 1:
            h = 0
        t = a[h]
        a[h] = 0
        a[0] = t - 1
        a[h + 1] += 1
        h += 1
        res.append(a.copy())
    return res


def onehot(x, num_class=None, dim=-1):
    if type(x) == np.ndarray:
        return onehot_numpy(x, num_class, dim)
    elif type(x) == torch.Tensor:
        return onehot_torch(x, num_class, dim)
    else:
        raise ValueError(
            f"{type(x)} is an invalid type, valid types are [np.array, torch.Tensor]"
        )


def onehot_torch(x, num_class=None, dim=-1):
    if num_class is None:
        num_class = num_class or int(x.max()) + 1
    x_onehot = torch.zeros(*tuple(x.size()), num_class).float()
    if torch.cuda.is_available():
        x = x.cuda()
        x_onehot = x_onehot.cuda()
    x = x.long().view(-1, 1)
    x_onehot.scatter_(dim, x, 1)
    return x_onehot


def onehot_numpy(x, num_class=None, dim=-1):
    num_class = num_class or int(x.max()) + 1
    y = np.zeros((*x.shape, num_class))
    i_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1, num_class)
    y[np.arange(x.size), x.astype("int")] = 1
    y = y.reshape((*i_shape, num_class))
    return y


def to_edge_index(g):
    def _to_edge_index(_g):
        assert issubclass(_g.__class__, nx.Graph)
        if not nx.is_directed(_g):
            _g = _g.to_directed()
        if _g.number_of_edges() == 0:
            return np.zeros((2, 0))
        else:
            return np.array(list(nx.to_edgelist(_g)))[:, :2].astype("int").T

    if isinstance(g, dict):
        edge_index = {}
        for k, v in g.items():
            edge_index[k] = _to_edge_index(v)
    else:
        edge_index = _to_edge_index(g)

    return edge_index


def collapse_networks(g):
    if not isinstance(g, dict):
        return g
    collapsed_g = nx.empty_graph()
    for k, v in g.items():
        edge_list = to_edge_index(v).T
        edge_attr = get_edge_attr(v)
        collapsed_g.add_edges_from(edge_list)
        for i, (u, v) in enumerate(edge_list):
            if "weight" in collapsed_g.edges[u, v]:
                collapsed_g.edges[u, v]["weight"] += edge_attr["weight"][i]
            else:
                collapsed_g.edges[u, v]["weight"] = edge_attr["weight"][i]
    return collapsed_g


def get_edge_weights(g):
    def _get_edge_weights(_g):
        if not nx.is_directed(_g):
            _g = _g.to_directed()
        edge_index = to_edge_index(_g).T
        weights = np.zeros((edge_index.shape[0], 1))
        for i, (u, v) in enumerate(edge_index):
            if "weight" in _g.edges[u, v]:
                weights[i] = _g.edges[u, v]["weight"]
            else:
                weights[i] = 1
        return weights

    if isinstance(g, dict):
        weights = {}
        for k, v in g.items():
            weights[k] = _get_edge_weights(v)
    else:
        weights = _get_edge_weights(g)

    return weights


def get_edge_attr(g, to_data=False):
    def _get_edge_attr(_g, to_data=False):
        if not nx.is_directed(_g):
            _g = _g.to_directed()

        edge_index = to_edge_index(_g).T
        attributes = {}
        n = edge_index.shape[0]
        for i, (u, v) in enumerate(edge_index):
            attr = _g.edges[u, v]
            for k, a in attr.items():
                if k not in attributes:
                    attributes[k] = np.zeros(n)
                attributes[k][i] = a
        if to_data:
            if len(attributes) > 0:
                return np.concatenate(
                    [v.reshape(-1, 1) for k, v in attributes.items()],
                    axis=-1,
                )
            else:
                return np.zeros((n, 0))
        return attributes

    if isinstance(g, dict):
        attributes = {}
        for k, v in g.items():
            attributes[k] = _get_edge_attr(v, to_data=to_data)
    else:
        attributes = _get_edge_attr(g, to_data=to_data)

    return attributes


def set_edge_attr(g, edge_attr):
    def _set_edge_attr(g, edge_attr):
        edge_index = to_edge_index(g).T
        for k, attr in edge_attr.items():
            for i, (u, v) in enumerate(edge_index):
                g.edges[u, v][k] = attr[i]
        return g

    if isinstance(g, dict):
        attributes = {}
        if edge_attr.keys != g.keys():
            edge_attr = {k: edge_attr for k in g.keys()}
        for k, v in g.items():
            g[k] = _set_edge_attr(v, edge_attr[k])
    else:
        g = _set_edge_attr(g, edge_attr)
    return g


def get_node_strength(g):
    def _get_node_strength(g):
        if not nx.is_directed(g):
            g = g.to_directed()

        strength = np.zeros(g.number_of_nodes())

        for u, v in g.edges():
            if "weight" in g.edges[u, v]:
                strength[u] += g.edges[u, v]["weight"]

        return strength

    if isinstance(g, dict):
        strength = {}
        for k, v in g.items():
            strength[k] = _get_node_strength(v)
    else:
        strength = _get_node_strength(g)
    return strength


def get_node_attr(g, to_data=False):
    def _get_node_attr(g, to_data=False):
        if not nx.is_directed(g):
            g = g.to_directed()

        n = g.number_of_nodes()
        attributes = {}

        for i, u in enumerate(g.nodes()):
            attr = g.nodes[u]
            for k, a in attr.items():
                if k not in attributes:
                    attributes[k] = np.zeros(n)
                attributes[k][i] = a
        if to_data:
            if len(attributes) > 0:
                return np.concatenate(
                    [v.reshape(-1, 1) for k, v in attributes.items()],
                    axis=-1,
                )
            else:
                return np.zeros((n, 0))
        return attributes

    if isinstance(g, dict):
        attributes = {}
        for k, v in g.items():
            attributes[k] = _get_node_attr(v, to_data=to_data)
    else:
        attributes = _get_node_attr(g, to_data=to_data)
    return attributes


def set_node_attr(g, node_attr):
    def _set_node_attr(g, node_attr):
        for k, attr in node_attr.items():
            for i in g.nodes():
                g.nodes[i][k] = attr[i]
        return g

    if isinstance(g, dict):
        if node_attr.keys != g.keys():
            node_attr = {k: node_attr for k in g.keys()}
        for k, v in g.items():
            g[k] = _set_node_attr(v, node_attr[k])
    else:
        g = _set_node_attr(g, node_attr)
    return g


def from_weighted_edgelist(edge_list, create_using=None):
    g = create_using or nx.Graph()
    for edge in edge_list:
        if len(edge) == 3:
            g.add_edge(int(edge[0]), int(edge[1]), weight=edge[2])
        else:
            g.add_edge(int(edge[0]), int(edge[1]), weight=1)
    return g


def loading_covid_data(experiment, path_to_covid, lag=1, lagstep=1, incidence=True):
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
        "inputs": DataCollection(name="inputs"),
        "targets": DataCollection(name="targets"),
        "networks": DataCollection(name="networks"),
    }
    inputs = np.zeros((X.shape[0] - (lag - 1) * lagstep, X.shape[1], num_states, lag))
    targets = np.zeros((Y.shape[0] - (lag - 1) * lagstep, Y.shape[1], num_states))
    for t in range(inputs.shape[0]):
        x = X[t : t + lag * lagstep : lagstep]
        y = Y[t + lag * lagstep]
        if incidence:
            x = x.reshape(*x.shape, 1)
            y = y.reshape(*y.shape, 1)
        x = np.transpose(x, (1, 2, 0))
        inputs[t] = x
        targets[t] = y
    data["inputs"].add(StateData(data=inputs))
    data["targets"].add(StateData(data=targets))
    data["networks"].add(NetworkData(data=networks))
    pop = data["networks"][0].data.node_attr["population"]
    experiment.dataset.data = data
    experiment.test_dataset = experiment.dataset.partition(
        type="cleancut", ti=335, tf=-1
    )
    experiment.partition_val_dataset()
    return experiment


def get_dataset_from_timeseries(ts, lag=1, lagstep=1):
    if ts.ndim == 3:
        num_steps, num_nodes, num_feats = ts.shape[0], ts.shape[1], ts.shape[2]
    elif ts.ndim == 2:
        num_steps, num_nodes, num_feats = ts.shape[0], ts.shape[1], 1
    inputs = np.zeros((num_steps - lag * lagstep, num_nodes, num_feats, lag))
    targets = np.zeros((num_steps - lag * lagstep, num_nodes, num_feats))

    for t in range(num_steps - lag * lagstep):
        x = ts[t : t + lag * lagstep : lagstep]
        inputs[t] = np.transpose(x, (1, 2, 0))
        targets[t] = ts[t + lag * lagstep]
    return inputs, targets
