from fastai.torch_core import *
from fastai.layers import *

__all__ = ['PCNet']


class PCNet(SequentialEx):
    def __init__(self, encoder:nn.Module, n_classes: int):
        # encoder has an output either of BxF
        pass
