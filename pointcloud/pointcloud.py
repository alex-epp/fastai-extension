from fastai.core import *
from fastai.torch_core import *
import pyntcloud

__all__ = ['open_ptcloud', 'open_ptmask']


TensorPtCloud = Tensor


class PtCloud(ItemBase):
    def __init__(self, pts: Tensor):
        self.pts = pts

    def clone(self):
        "Mimic the behaviour of torch.clone for `PtCloud` objects."
        return self.__class__(self.pts.clone())

    @property
    def shape(self) -> Tuple[int, int]: return self._pts.shape
    @property
    def size(self) -> int: return self._pts.shape[0]
    @property
    def device(self) -> torch.device: return self._pts.device

    def __repr__(self): return f'{self.__class__.__name__} {tuple(self.shape)}'

    # TODO: transforms

    # TODO: show





def open_ptcloud(fn: PathOrStr, after_open: Callable = None):
    raise NotImplementedError()


def open_ptmask(fn: PathOrStr, after_open: Callable = None):
    raise NotImplementedError()


if __name__ == '__live_coding__':
    print('Hello, world!')
