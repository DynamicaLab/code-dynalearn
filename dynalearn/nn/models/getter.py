from torch_geometric.nn import GATConv, SAGEConv, GCNConv, GraphConv, GINConv
from torch.nn import Module, Linear, Sequential, ReLU, Dropout
from .dgat import DynamicsGATConv
from .util import Reshape, MultiplexLayer
from ..activation import get as get_activation


class IndependentGNN(Module):
    def __init__(self, fin, fout, bias=True):
        Module.__init__(self)
        self.fin = fin
        self.fout = fout
        self.bias = bias

        self.layer = Linear(fin, fout, bias=bias)

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x)

    def reset_parameters(self):
        self.layer.reset_parameters()


class FullyConnectedGNN(Module):
    def __init__(
        self, fin, fout, num_nodes, agg_in=4, agg_out=4, activation="relu", bias=True
    ):
        Module.__init__(self)
        self.fin = fin
        self.fout = fout
        self.num_nodes = num_nodes
        self.agg_in = agg_in
        self.agg_out = agg_out
        self.bias = bias
        self.activation = get_activation(activation)

        self.in_layer = Linear(fin, agg_in, bias=bias)
        self.layer = Linear(
            self.agg_in * self.num_nodes, self.agg_out * self.num_nodes, bias=bias
        )
        self.out_layer = Linear(agg_out, fout, bias=bias)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.in_layer(x)
        x = x.view(self.num_nodes * self.agg_in)
        x = self.activation(x)
        x = self.layer(x)
        x = self.activation(x)
        x = x.view(self.num_nodes, self.agg_out)
        return self.out_layer(x)

    def reset_parameters(self):
        self.layer.reset_parameters()


class KapoorConv(Module):
    def __init__(self, in_channels, out_channels, config):
        Module.__init__(self)
        hidden_channels = in_channels
        self.layer1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.activation = ReLU()
        self.layer2 = GCNConv(hidden_channels, out_channels)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.dropout(x)
        x = self.layer1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, edge_index)
        return x

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


class CustomGNN(Module):
    def __init__(self, gnn_layer):
        Module.__init__(self)
        self.layer = gnn_layer

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)

    def reset_parameters(self):
        self.layer.reset_parameters()


def get_GATConv(in_channels, out_channels, config):
    return CustomGNN(
        GATConv(
            in_channels,
            out_channels,
            heads=config.heads,
            concat=config.concat,
            bias=config.bias,
        )
    )


def get_SAGEConv(in_channels, out_channels, config):
    # if config.concat:
    #     out_channels *= config.heads
    return CustomGNN(SAGEConv(in_channels, out_channels, bias=config.bias))


def get_GCNConv(in_channels, out_channels, config):
    # if config.concat:
    #     out_channels *= config.heads
    return CustomGNN(
        GCNConv(
            in_channels,
            out_channels,
            bias=config.bias,
        )
    )


def get_MaxGraphConv(in_channels, out_channels, config):
    # if config.concat:
    #     out_channels *= config.heads
    return CustomGNN(GraphConv(in_channels, out_channels, bias=config.bias, aggr="max"))


def get_MeanGraphConv(in_channels, out_channels, config):
    # if config.concat:
    #     out_channels *= config.heads
    return CustomGNN(
        GraphConv(in_channels, out_channels, bias=config.bias, aggr="mean")
    )


def get_AddGraphConv(in_channels, out_channels, config):
    # if config.concat:
    #     out_channels *= config.heads
    return CustomGNN(GraphConv(in_channels, out_channels, bias=config.bias, aggr="add"))


def get_KapoorConv(in_channels, out_channels, config):
    return CustomGNN(KapoorConv(in_channels, out_channels, config))


def get_DynamicsGATConv(in_channels, out_channels, config):
    if "edge_channels" not in config.__dict__:
        in_edge = 0
    else:
        in_edge = config.edge_channels[-1]

    if "edge_gnn_channels" not in config.__dict__:
        out_edge = 0
    else:
        out_edge = config.edge_gnn_channels

    return DynamicsGATConv(
        in_channels,
        out_channels,
        heads=config.heads,
        concat=config.concat,
        bias=config.bias,
        edge_in_channels=in_edge,
        edge_out_channels=out_edge,
        self_attention=config.self_attention,
    )


def get_Independent(in_channels, out_channels, config):
    return IndependentGNN(in_channels, out_channels, bias=config.bias)


def get_FullyConnected(in_channels, out_channels, config):
    return FullyConnectedGNN(
        in_channels, out_channels, num_nodes=config.num_nodes, bias=config.bias
    )


__gnn_layers__ = {
    "GATConv": get_GATConv,
    "SAGEConv": get_SAGEConv,
    "GCNConv": get_GCNConv,
    "AddGraphConv": get_AddGraphConv,
    "MeanGraphConv": get_MeanGraphConv,
    "MaxGraphConv": get_MaxGraphConv,
    "KapoorConv": get_KapoorConv,
    "DynamicsGATConv": get_DynamicsGATConv,
    "IndependentGNN": get_Independent,
    "FullyConnectedGNN": get_FullyConnected,
}


def get(config):
    fin = config.in_channels[-1] + config.node_channels[-1]
    fout = config.gnn_channels
    name = config.gnn_name
    if name in __gnn_layers__:
        template = lambda: __gnn_layers__[name](fin, fout, config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__gnn_layers__.keys())}"
        )
    if "is_multiplex" in config.__dict__ and config.is_multiplex:
        layer = MultiplexLayer(template, config.network_layers, merge="mean")
    else:
        layer = template()
    return layer
