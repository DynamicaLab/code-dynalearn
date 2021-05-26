import torch

from .transformer import TransformerDict, CUDATransformer
from .normalizer import InputNormalizer, TargetNormalizer, NetworkNormalizer
from dynalearn.util import get_node_attr


class BatchNormalizer(TransformerDict):
    def __init__(
        self,
        input_size=0,
        target_size=0,
        node_size=0,
        edge_size=0,
        layers=None,
        auto_cuda=True,
    ):
        transformer_dict = {"t_cuda": CUDATransformer()}
        if input_size is not None:
            transformer_dict["t_inputs"] = InputNormalizer(
                input_size, auto_cuda=auto_cuda
            )
        else:
            transformer_dict["t_inputs"] = CUDATransformer()

        if target_size is not None:
            transformer_dict["t_targets"] = TargetNormalizer(target_size)
        else:
            transformer_dict["t_targets"] = CUDATransformer()

        transformer_dict["t_networks"] = NetworkNormalizer(
            node_size, edge_size, layers=layers, auto_cuda=auto_cuda
        )

        TransformerDict.__init__(self, transformer_dict)

    def forward(self, data):
        (x, g), y, w = data
        x = self["t_inputs"].forward(x)
        g = self["t_networks"].forward(g)
        y = self["t_targets"].forward(y)
        w = self["t_cuda"].forward(w)
        return (x, g), y, w

    def backward(self, data):
        (x, g), y, w = data
        x = self["t_inputs"].backward(x)
        g = self["t_networks"].backward(g)
        y = self["t_targets"].backward(y)
        w = self["t_cuda"].backward(w)
        return (x, g), y, w
