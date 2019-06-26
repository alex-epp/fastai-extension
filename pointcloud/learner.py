from . import models
from fastai.basic_data import *
from fastai.core import *
from fastai.basic_train import *
from fastai.torch_core import *
from typing import *

import torch.backends.cudnn

__all__ = ['pcn_learner']


def pcn_learner(data: DataBunch, arch: Callable, pretrained: bool = True, input_channels=None,
                **kwargs: Any) -> Learner:
    body = arch(pretrained=pretrained, input_channels=input_channels)
    model = to_device(models.PCNet(body), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split((model.encoder, model.decoder))
    if pretrained: learn.freeze()
    apply_init(model.decoder, nn.init.kaiming_normal_)
    return learn
