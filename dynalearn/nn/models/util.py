import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter, Linear, Sequential, Module
from torch.nn.init import kaiming_normal_
from torch_geometric.nn.inits import glorot, zeros
from dynalearn.nn.activation import get as get_activation

__rnn_layers__ = {
    "RNN": lambda config: torch.nn.RNN(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        nonlinearity=config.activation,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
    "LSTM": lambda config: torch.nn.LSTM(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
    "GRU": lambda config: torch.nn.GRU(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
}


class ParallelLayer(nn.Module):
    def __init__(self, template, keys, merge="None"):
        nn.Module.__init__(self)
        self.template = template
        self.keys = keys
        self.merge = merge

        for k in self.keys:
            setattr(self, f"layer_{k}", template())

    def forward(self, x, **kwargs):
        y = None
        for k in self.keys:
            yy = getattr(self, f"layer_{k}")(x[k], **kwargs)
            if y is None:
                if isinstance(yy, tuple):
                    y = [{k: yyy} for yyy in yy]
                else:
                    y = {k: yy}
            else:
                if isinstance(yy, tuple):
                    for i, yyy in enumerate(yy):
                        y[i][k] = yyy
                else:
                    y[k] = yy
        if self.merge == "concat":
            return self._merge_concat_(y)
        elif self.merge == "mean":
            return self._merge_mean_(y)
        elif self.merge == "sum":
            return self._merge_sum_(y)
        else:
            if isinstance(y, list):
                return tuple(y)
            else:
                return y

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        for k in self.keys:
            layer = getattr(self, f"layer_{k}")
            if isinstance(layer, Sequential):
                for l in layer:
                    if type(l) == Linear:
                        initialize_inplace(l.weight)
                        if l.bias is not None:
                            l.bias.data.fill_(0)
            else:
                try:
                    layer.reset_parameters()
                except:
                    pass

    def __repr__(self):
        name = self.template().__class__.__name__
        return "{}({}, heads={})".format(
            self.__class__.__name__,
            name,
            len(self.keys),
        )

    def _merge_mean_(self, y):
        if isinstance(y, list):
            out = (
                torch.cat(
                    [yy[k].view(*yy[k].shape, 1) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            )
            return [torch.mean(yy, axis=-1) for yy in out]
        else:
            out = torch.cat(
                [y[k].view(*y[k].shape, 1) for k in self.keys],
                axis=-1,
            )
            return torch.mean(out, axis=-1)

    def _merge_sum_(self, y):
        if isinstance(y, list):
            out = (
                torch.cat(
                    [yy[k].view(*yy[k].shape, 1) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            )
            return [torch.sum(yy, axis=-1) for yy in out]
        else:
            out = torch.cat(
                [y[k].view(*y[k].shape, 1) for k in self.keys],
                axis=-1,
            )
            return torch.sum(out, axis=-1)

    def _merge_concat_(self, y):
        if isinstance(y, list):
            return [
                torch.cat(
                    [yy[k].view(*yy[k].shape) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            ]
        else:
            return torch.cat(
                [y[k].view(*y[k].shape) for k in self.keys],
                axis=-1,
            )


class MultiplexLayer(ParallelLayer):
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if edge_attr is None:
            y = {
                k: getattr(self, f"layer_{k}")(x, edge_index[k], **kwargs)
                for k in self.keys
            }
        else:
            y = None
            for k in self.keys:

                yy = getattr(self, f"layer_{k}")(
                    x, edge_index[k], edge_attr[k], **kwargs
                )
                if y is None:
                    if isinstance(yy, tuple):
                        y = [{k: yyy} for yyy in yy]
                    else:
                        y = {k: yy}
                else:
                    if isinstance(yy, tuple):
                        for i, yyy in enumerate(yy):
                            y[i][k] = yyy
                    else:
                        y[k] = yy
        if isinstance(y, list):
            y = y[0]
        if self.merge == "concat":
            return self._merge_concat_(y)
        elif self.merge == "mean":
            return self._merge_mean_(y)
        elif self.merge == "sum":
            return self._merge_sum_(y)
        else:
            return y


class MultiHeadLinear(nn.Module):
    def __init__(self, num_channels, heads=1, bias=False):
        nn.Module.__init__(self)
        self.num_channels = num_channels
        self.heads = heads

        self.weight = Parameter(torch.Tensor(heads, num_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(heads))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = (x * self.weight).sum(dim=-1)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self):
        return "{}({}, heads={})".format(
            self.__class__.__name__, self.num_channels, self.heads
        )


class LinearLayers(nn.Module):
    def __init__(self, channels, activation="relu", bias=True, **kwargs):
        nn.Module.__init__(self)
        self.channels = channels
        self.activation = get_activation(activation)
        self.bias = bias
        self.template = lambda fin, fout: nn.Linear(fin, fout, bias=self.bias)
        self.build()

    def build(self):
        layers = []
        for f in zip(self.channels[:-1], self.channels[1:]):
            layers.append(self.template(*f))
            layers.append(self.activation)
        self.layers = Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            n = x.shape[0]
            x = x.view(n, -1)
        return self.layers(x)

    def reset_parameters(self, initialize_inplace=None):
        reset_layer(self.layers, initialize_inplace=initialize_inplace)


class RNNLayers(nn.Module):
    def __init__(
        self,
        channels,
        activation="relu",
        layer="rnn",
        bidirectional=False,
        bias=True,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.channels = channels
        self.activation = get_activation(activation)
        self.bidirectional = bidirectional
        self.bias = bias
        if layer == "rnn":
            self.template = lambda fin, fout, num_layers: nn.RNN(
                fin, fout, num_layers, bidirectional=self.bidirectional, bias=self.bias
            )
        elif layer == "lstm":
            self.template = lambda fin, fout, num_layers: nn.LSTM(
                fin, fout, num_layers, bidirectional=self.bidirectional, bias=self.bias
            )
        elif layer == "gru":
            self.template = lambda fin, fout, num_layers: nn.GRU(
                fin, fout, num_layers, bidirectional=self.bidirectional, bias=self.bias
            )
        elif issubclass(layer.__class__, nn.Module):
            self.template = layer
        else:
            raise TypeError(
                f"Invalid type, expected `[str, Module]` but received `{type(layer)}`."
            )
        self.build()

    def build(self):
        # self.layers = []
        # for f in zip(self.channels[:-1], self.channels[1:]):
        #     self.layers.append(self.template(*f))
        # self.layers = nn.ModuleList(self.layers)
        fin, fout = self.channels[0], self.channels[-1]
        num_layers = len(self.channels) - 1
        self.layers = self.template(fin, fout, num_layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        x  # [batch, features, timestamps]
        x = torch.transpose(x, 0, 1)  # [features, batch, timestamps]
        x = torch.transpose(x, 0, 2)  # [timestamps, batch, features]
        # for l in self.layers:
        #     x = l(x)[0]
        x = self.layers(x)
        if isinstance(x, tuple):
            x = x[0]
        return self.activation(x[-1])

    def reset_parameters(self):
        # for l in self.layers:
        #     l.reset_parameters()
        self.layers.reset_parameters()


class Reshape(nn.Module):
    def __init__(self, shape):
        self.shape = shape
        nn.Module.__init__(self)

    def forward(self, x):
        return x.view(*self.shape)


class Transpose(nn.Module):
    def __init__(self, ax1, ax2):
        self.ax1 = ax1
        self.ax2 = ax2
        nn.Module.__init__(self)

    def forward(self, x):
        return torch.transpose(x, self.ax1, self.ax2)


class Troncate(nn.Module):
    def __init__(self, ax):
        assert isinstance(
            ax, (int, type(None))
        ), f"Invalid type: expected types `[int, NoneType]` but received `{type(x)}` for `ax`."
        self.ax = ax
        nn.Module.__init__(self)

    def forward(self, x):
        return x[self.ax]


def get_in_layers(config):
    if "type" not in config.__dict__:
        config.type = "linear"
    if config.type == "linear":
        in_channels = [config.lag * config.in_size] + config.in_channels
        return LinearLayers(in_channels, config.in_activation, bias=config.bias)
    elif config.type == "rnn" or config.type == "lstm" or config.type == "gru":
        in_channels = [config.in_size] + config.in_channels
        return RNNLayers(
            in_channels,
            config.in_activation,
            bias=config.bias,
            layer=config.type,
        )
    else:
        raise ValueError(
            f"Invalid layer: expected `['linear', 'lstm', 'lstm', 'gru']` but received `{config.type}`."
        )


def get_out_layers(config):
    if config.concat and config.gnn_name in ["DynamicsGATConv", "GATConv"]:
        out_layer_channels = [config.heads * config.gnn_channels] + config.out_channels
    else:
        out_layer_channels = [config.gnn_channels] + config.out_channels
    layers = LinearLayers(out_layer_channels, config.out_activation, bias=config.bias)
    layers = Sequential(
        layers,
        Linear(
            config.out_channels[-1],
            config.out_size,
            bias=config.bias,
        ),
        get_activation(config.out_act),
    )
    return layers


def reset_layer(layer, initialize_inplace=None):
    if initialize_inplace is None:
        initialize_inplace = kaiming_normal_
    assert isinstance(layer, Module)

    if isinstance(layer, Sequential):
        for l in layer:
            if type(l) == Linear:
                initialize_inplace(l.weight)
                if l.bias is not None:
                    l.bias.data.fill_(0)
    else:
        layer.reset_parameters()
