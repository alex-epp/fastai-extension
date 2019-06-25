from . import models
from fastai.basic_data import *
from fastai.core import *
from fastai.basic_train import *
from fastai.torch_core import *
from typing import *

import torch.backends.cudnn

__all__ = ['pcn_learner']


def _default_split(m: nn.Module): return m[1],
def _pcn_split(m: nn.Module): return m.encoder, m.decoder


_default_meta = {'cut': None, 'split': _default_split}
_pointnet2_cls_meta = {'cut': -1, 'split': _pcn_split}
_pointnet2_seg_meta = {'cut': -1, 'split': _pcn_split}


model_meta = {
    models.pointnet2_msg_cls: {**_pointnet2_cls_meta},
    models.pointnet2_ssg_cls: {**_pointnet2_cls_meta},
    models.pointnet2_msg_seg: {**_pointnet2_seg_meta},
    models.pointnet2_ssg_seg: {**_pointnet2_seg_meta},
}


def pointnet_config(arch):
    torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)


def create_body(arch: Callable, pretrained: bool = True, cut: Union[int, Callable] = None):
    model: nn.Module = arch(pretrained)
    cut = ifnone(cut, pointnet_config(arch)['cut'])
    if isinstance(cut, int): return nn.Sequential(*list(model.children()))[:cut]
    elif isinstance(cut, Callable): return cut(model)
    else: raise NameError('cut must either be integer or a function')


def pcn_learner(data: DataBunch, arch: Callable, pretrained: bool = True,
                cut:Union[int, Callable] = None, split_on: SplitFuncOrIdxList = None,
                **kwargs: Any) -> Learner:
    meta = pointnet_config(arch)
    body = create_body(arch, pretrained=pretrained, cut=cut)
    model = to_device(models.PCNet(body), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn
