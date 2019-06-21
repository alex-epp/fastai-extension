from fastai.core import *
from fastai.torch_core import *
import pyntcloud

__all__ = ['split_ptcloud']


def split_ptcloud(pts: pyntcloud.PyntCloud, cell_size: Union[float, Iterable]
                  ) -> List[pyntcloud.PyntCloud]:
    cell_size = listify(cell_size, 3)

    df = pts.points.assign(
        group_x=pts.points.x // cell_size[0],
        group_y=pts.points.y // cell_size[1],
        group_z=pts.points.z // cell_size[2],
    )
    return [
        pyntcloud.PyntCloud(group.drop(columns=['group_x', 'group_y', 'group_z']))
        for _, group in df.groupby(by=['group_x', 'group_y', 'group_z'], as_index=False)
    ]


# __all__ = ['open_ptcloud', 'open_ptmask', 'PtCloud']
#
#
# TensorPtCloud = Tensor
#
#
# class PtCloud(ItemBase):
#     def __init__(self, pts: Tensor):
#         self.data = pts
#
#     def clone(self):
#         "Mimic the behaviour of torch.clone for `PtCloud` objects."
#         return self.__class__(self.data.clone())
#
#     @property
#     def shape(self) -> Tuple[int, int]: return self.data.shape
#     @property
#     def size(self) -> int: return self.data.shape[0]
#     @property
#     def device(self) -> torch.device: return self.data.device
#
#     def __repr__(self): return f'{self.__class__.__name__} {tuple(self.shape)}'
#
#
#     # TODO: transforms
#
#     # TODO: show
#
#
#
#
#
# def open_ptcloud(fn: PathOrStr,  after_open: Callable = None) -> PtCloud:
#     raise NotImplementedError()
#
#
# def open_ptmask(fn: PathOrStr, after_open: Callable = None) -> PtCloud:
#     raise NotImplementedError()
#
#
# if __name__ == '__live_coding__':
#     print('Hello, world!')
