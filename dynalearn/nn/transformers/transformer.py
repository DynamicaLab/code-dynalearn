import torch

from abc import abstractmethod
from dynalearn.util import get_edge_attr, get_node_attr


class Transformer(torch.nn.Module):
    def __init__(self, name):
        torch.nn.Module.__init__(self)
        self.name = name
        self.is_empty = False
        if torch.cuda.is_available():
            self = self.cuda()

    @abstractmethod
    def forward(self, x):
        raise NotImplemented()

    @abstractmethod
    def backward(self, x):
        raise NotImplemented()

    def setUp(self, dataset):
        for method in dir(self):
            if method[: len("_setUp_")] == "_setUp_":
                label = method[len("_setUp_") :]
                m = getattr(self, method)(dataset)
                if isinstance(m, torch.Tensor) and torch.cuda.is_available():
                    m = m.cuda()
                setattr(self, f"{self.name}_{label}", m)


class IdentityTransformer(Transformer):
    def __init__(self):
        Transformer.__init__(self, "identity")

    def forward(self, x):
        return x

    def backward(self, x):
        return x


class CUDATransformer(Transformer):
    def __init__(self):
        Transformer.__init__(self, "cuda")

    def forward(self, x):
        if torch.cuda.is_available():
            if isinstance(x, dict):
                for k in x.keys():
                    assert isinstance(v, torch.Tensor)
                    x[k] = v.cuda()
            else:
                assert isinstance(x, torch.Tensor)
                x = x.cuda()
        return x

    def backward(self, x):
        if isinstance(x, dict):
            for k, v in x.items():
                assert isinstance(v, torch.Tensor)
                x[k] = v.cpu()
        else:
            assert isinstance(x, torch.Tensor)
            x = x.cpu()
        return x


class TransformerDict(torch.nn.ModuleDict):
    def __init__(self, transformers={}):
        torch.nn.Module.__init__(self)
        assert isinstance(transformers, dict)
        for t in transformers.values():
            assert issubclass(
                t.__class__, Transformer
            ), f"{t.__class__.__name__} is not a subclass of Transformer."
        torch.nn.ModuleDict.__init__(self, modules=transformers)

    def forward(self, x, key):
        if key in self.keys():
            return self[key].forward(x)
        else:
            return x

    def backward(self, x, key):
        if key in self.keys():
            return self[key].backward(x)
        else:
            return x

    def setUp(self, dataset):
        for t in self.values():
            t.setUp(dataset)
