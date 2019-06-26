from fastai.basic_data import *
from fastai.core import *
from fastai.data_block import *
from fastai.layers import *
from fastai.torch_core import *
import torch

from .pointcloud import *
from .losses import *
import pyntcloud

__all__ = ['PtCloudDataBunch', 'PtCloudSegmentationList', 'PtCloudUpsampleList']

# TODO: set to all file types loadable with the point-cloud library I choose
ptcloud_extensions = ['.las', '.laz', '.ply']

TensorPtCloud = Tensor


def normalize(x: TensorPtCloud, mean: FloatTensor, std: FloatTensor):
    masked_out = (x == 0).all(dim=-1)
    normed = (x - mean[None, None, :]) / std[None, None, :]
    normed[masked_out] = 0
    return normed


def _normalize_batch(b: Tuple[Tensor, Tensor], mean: FloatTensor, std: FloatTensor,
                     do_x: bool = True, do_y: bool = True):
    x, y = b
    mean, std = mean.to(x.device), std.to(x.device)
    if do_x:
        x = torch.cat((x[..., :3], normalize(x[..., 3:], mean, std)), dim=-1)
    if do_y and len(y.shape) == 3:
        y = torch.cat((y[..., :3], normalize(y[..., 3:], mean, std)), dim=-1)
    return x, y


def denormalize(x: TensorPtCloud, mean: FloatTensor, std: FloatTensor, do_x: bool = True, do_y: bool = True):
    assert do_y is False

    if do_x:
        x = x.cpu().float()
        return torch.cat((x[..., :3],
                          x[..., 3:] * std[None, None, ...] + mean[None, None, ...]),
                         dim=-1)
    else:
        return x.cpu()


def normalize_funcs(mean: FloatTensor, std: FloatTensor, do_x: bool, do_y: bool):
    mean, std = tensor(mean), tensor(std)
    return (partial(_normalize_batch, mean=mean, std=std, do_x=do_x, do_y=do_y),
            partial(denormalize, mean=mean, std=std, do_x=do_x, do_y=do_y))


class PtCloudDataBunch(DataBunch):
    "DataBunch suitable for point-cloud processing."

    def batch_stats(self,
                    funcs: Collection[Callable] = None,
                    ds_type: DatasetType = DatasetType.Train
                    ):
        funcs = ifnone(funcs,[torch.mean, torch.std])
        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()

        n_features = x.shape[2] - 3
        if n_features <= 0: return None

        masked_out = (x == 0).all(dim=-1)
        x = x[~masked_out][:, 3:].transpose(0, 1)

        return [func(x, 1) for func in funcs]

    def normalize(self,
                  stats: Collection[Tensor] = None,
                  do_x: bool = True,
                  do_y: bool = False
                  ) -> 'PtCloudDataBunch':
        if getattr(self, 'norm', False):
            raise Exception('Can not call normalize twice')

        self.stats = ifnone(stats, self.batch_stats())
        if self.stats is not None:
            self.norm, self.denorm = normalize_funcs(*self.stats, do_x=do_x, do_y=do_y)
            self.add_tfm(self.norm)
        return self


class PtCloudList(ItemList):
    "ItemList suitable for computer vision."
    _bunch = PtCloudDataBunch

    def __init__(self, items: Iterator,
                 *args,
                 pt_clouds=None,
                 features=None,
                 **kwargs):
        if pt_clouds is None:
            pt_clouds = [pyntcloud.PyntCloud.from_file(str(f)) for f in items]
            items = range_of(pt_clouds)

        super().__init__(items, *args, **kwargs)

        self.features = listify(features)
        self.pt_clouds = pt_clouds
        self.copy_new.extend(['features', 'pt_clouds'])
        # TODO: ImageList sets 'self.c' here. Why?

    def open(self, i):
        return PtCloudItem.from_ptcloud(self.pt_clouds[i],
                                        ['x', 'y', 'z'] + self.features)

    def get(self, i):
        i = super().get(i)
        return self.open(i)

    @classmethod
    def from_folder(cls,
                    path: PathOrStr = '.',
                    extensions: Collection[str] = None,
                    **kwargs
                    ) -> 'PtCloudList':
        extensions = ifnone(extensions, ptcloud_extensions)
        return super().from_folder(path=path, extensions=extensions, **kwargs)

    def reconstruct(self, t: Tensor) -> PtCloudItem:
        return PtCloudItem(t.float())

    # TODO: show

    def filter(self, filter_, *, from_item_lists=False):
        if from_item_lists:
            raise Exception('Can\'t use filter after splitting data.')

        pt_clouds = list(filter(filter_, self.pt_clouds))
        return self.new(np.asarray(range_of(pt_clouds)), pt_clouds=pt_clouds)

    def chunkify(self, chunk_size: Union[int, Iterable] = 1, *, from_item_lists=False):
        if from_item_lists:
            raise Exception('Can\'t use chunkify after splitting data.')
        pts = []
        for p in self.pt_clouds:
            pts.extend(ptcloud_split(p, chunk_size))

        return self.new(range_of(pts), pt_clouds=pts)

    def voxel_sample(self, voxel_size: Union[float, Iterable] = 0.1,
                     agg='intensity', *, from_item_lists=False):
        if from_item_lists:
            raise Exception('Can\'t use voxel_sample after splitting data.')

        pt_clouds = [ptcloud_voxel_sample(p, voxel_size, agg) for p in self.pt_clouds]
        return self.new(self.items, pt_clouds=pt_clouds)

    def norm_xyz(self, scale=None, *, from_item_lists=False):
        if from_item_lists:
            raise Exception('Can\'t use normalize after splitting data.')
        pt_clouds = [ptcloud_norm_xyz(p, scale=scale) for p in self.pt_clouds]
        return self.new(self.items, pt_clouds=pt_clouds)


class PtCloudSegmentationProcessor(PreProcessor):
    "`PreProcessor` that stores the classes for segmentation."
    def __init__(self, ds: ItemList): self.classes = ds.classes
    def process(self, ds: ItemList): ds.classes, ds.c = self.classes, len(self.classes)


class PtCloudSegmentationLabelList(PtCloudList):
    "`ItemList` for point-cloud segmentation masks."
    _processor = PtCloudSegmentationProcessor

    def __init__(self,
                 items: Iterator,
                 classes: Collection = None,
                 label_field: str = 'classification',
                 **kwargs
                 ):
        super().__init__(items, **kwargs)
        self.copy_new.extend(['classes', 'label_field'])
        self.classes, self.label_field = classes, label_field
        self.loss_func = MaskedFlattenedLoss(nn.CrossEntropyLoss, axis=-1)

    def open(self, i):
        return PtCloudSegmentItem.from_ptcloud(self.pt_clouds[i],
                                               self.label_field)

    def analyze_pred(self, pred:Tensor):
        return pred.argmax(dim=1)[None]

    def reconstruct(self, t: Tensor):
        return PtCloudSegmentItem(t)


class PtCloudSegmentationList(PtCloudList):
    "`ItemList suitable for point cloud segmentation tasks."
    _label_cls = PtCloudSegmentationLabelList

    def label_from_field(self, label_field: str = 'classification', **kwargs):
        return self._label_from_list(self.items, pt_clouds=self.pt_clouds,
                                     label_field=label_field, **kwargs)


class PtCloudUpsampleProcessor(PreProcessor):
    "`PreProcessor` that stores the classes for segmentation."
    def __init__(self, ds: ItemList): self.classes = ds.classes
    def process(self, ds: ItemList): ds.classes, ds.c = self.classes, len(self.classes)


class PtCloudUpsampleLabelList(PtCloudList):
    "`ItemList` for point-cloud segmentation masks."
    _processor = None # PtCloudUpsampleProcessor

    def __init__(self,
                 items: Iterator,
                 **kwargs
                 ):
        super().__init__(items, **kwargs)
        self.loss_func = ChamferDistance()

    def open(self, i):
        return PtCloudUpsampledItem.from_ptcloud(self.pt_clouds[i], self.features)

    def analyze_pred(self, pred:Tensor):
        return pred.argmax(dim=1)[None]

    def reconstruct(self, t: Tensor):
        return PtCloudSegmentItem(t)


class PtCloudUpsampleList(PtCloudList):
    "`ItemList suitable for point cloud upsampling tasks."
    _label_cls = PtCloudUpsampleLabelList

    def label(self, downsample_cellsize=0.2, target_features=None, downsample_agg='intensity', **kwargs):
        ll = self._label_from_list(self.items,
                                   pt_clouds=self.pt_clouds,
                                   features=['x', 'y', 'z'] + listify(target_features),
                                   **kwargs)
        self.pt_clouds = [ptcloud_voxel_sample(pts, downsample_cellsize, downsample_agg) for pts in self.pt_clouds]
        return ll
