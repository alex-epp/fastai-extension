"Point cloud transformations for data augmentation. All transforms are done at the tensor level"
from fastai.torch_core import *
from random import Random

from .pointcloud import *

__all__ = ['sample', 'dropout', 'jitter_xyz', 'rotate_x', 'rotate_y', 'rotate_z',
           'get_transforms']

RandomSeed = partial(uniform_int, 0, 100000)


def _sample(n, k, *, seed: RandomSeed):
    return Random(seed).choices(range(n), k=k)
sample = PtCloudIdxTransform(_sample, order=0)


def _dropout(n, drop_pct: uniform, seed: RandomSeed):
    return Random(seed).sample(range(n), k=int(n * drop_pct))
dropout = PtCloudMaskOutTransform(_dropout, order=1)


def _jitter_xyz(xyz, magnitude=0.01, clamp=(-0.03, 0.03)):
    return xyz + torch.clamp(torch.rand_like(xyz) * magnitude, *clamp)
jitter_xyz = PtCloudXYZTransform(_jitter_xyz, order=2)


def _rotate_x(degrees: uniform):
    angle = degrees * math.pi / 180
    return [[1.,  0.,         0.        ],
            [0.,  cos(angle), sin(angle)],
            [0., -sin(angle), cos(angle)]]
rotate_x = PtCloudAffineTransform(_rotate_x, order=4)


def _rotate_y(degrees: uniform):
    angle = degrees * math.pi / 180
    return [[cos(angle), 0., -sin(angle)],
            [0.,         1.,   0.       ],
            [sin(angle), 0.,  cos(angle)]]
rotate_y = PtCloudAffineTransform(_rotate_y, order=4)


def _rotate_z(degrees: uniform):
    angle = degrees * math.pi / 180
    return [[ cos(angle), sin(angle), 0.],
            [-sin(angle), cos(angle), 0.],
            [ 0.        , 0.        , 1.]]
rotate_z = PtCloudAffineTransform(_rotate_z, order=4)


def get_transforms(n_sample=1024,
                   max_dropout_pct=0.7,
                   jitter=0.01,
                   jitter_clamp=(-0.03, 0.03),
                   max_rotate=180,
                   xtra_tfms=None):
    res = []
    if n_sample: res.append(sample(k=n_sample))
    if max_dropout_pct: res.append(dropout(drop_pct=(0, max_dropout_pct)))
    if jitter: res.append(jitter_xyz(magnitude=jitter, clamp=jitter_clamp))
    if max_rotate: res.append(rotate_z(degrees=(-max_rotate, max_rotate)))
    #      train                   , valid
    return res + listify(xtra_tfms), [res[0]] if n_sample else []
