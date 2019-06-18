# This file currently assumes a 1:1 relationship between point-cloud files
# and examples, which would make duplicating some papers needlessly complicated.
# Perhaps the PtcloudList could take some kind of point-cloud example generator?
# TODO: implement more flexible file-example relationship

from fastai.basic_data import *
from fastai.core import *
from fastai.data_block import *
from fastai.layers import *
from fastai.torch_core import *

from .pointcloud import *

__all__ = ['PtCloudDataBunch']

# TODO: set to all file types loadable with the point-cloud library I choose
ptcloud_extensions = ['.las', '.laz']


class PtCloudDataBunch(DataBunch):
    "DataBunch suitable for point-cloud processing."
    pass


class PtCloudList(ItemList):
    "`PtCloudList` suitable for point-cloud processing."
    _bunch = PtCloudDataBunch
    def __init__(self, *args, after_open:Callable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_open = after_open
        # TODO: add pointcloud file configuration here

    def open(self, fn):
        "Open point-cloud in `fn`, subclass and overwrite for custom behavior."
        return open_ptcloud(fn, after_open=self.after_open)

    def get(self, i):
        fn = super().get(i)
        res = self.open(fn)
        # TODO: ImageList stores size here. Find out why.
        return res

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, **kwargs)->ItemList:
        extensions = ifnone(extensions, ptcloud_extensions)
        return super().from_folder(path=path, extensions=extensions, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='', **kwargs)->'ItemList':
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."
        suffix = suffix or ''
        res = super().from_df(df, path=path, cols=cols, **kwargs)
        pref = f'{res.path}{os.path.sep}'
        if folder is not None: pref += f'{folder}{os.path.sep}'
        res.items = np.char.add(np.char.add(pref, res.items.astype(str)), suffix)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, header:str='infer', **kwargs)->'ItemList':
        "Get the filenames in `path/csv_name` opened with `header`."
        path = Path(path)
        df = pd.read_csv(path/csv_name, header=header)
        return cls.from_df(df, path=path, **kwargs)

    def reconstruct(self, t:Tensor): raise NotImplementedError()

class ptSegmentationProcessor(PreProcessor):
    "`PreProcessor` that stores the classes for point-cloud segmentation."
    def __init__(self, ds:ItemList): self.classes = ds.classes
    def process(self, ds:ItemList):  ds.classes,ds.c = self.classes,len(self.classes)

# NOTE: assumes point-cloud is stored as BxFxN
class ptSegmentationLabelList(PtCloudList):
    "`ItemList` for point-cloud segmentation masks."
    _processor=ptSegmentationProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        # TODO: allow other losses, e.g. dice
        self.classes,self.loss_func = classes,CrossEntropyFlat(axis=1)

    def open(self, fn): return open_ptmask(fn)
    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]
    def reconstruct(self, t:Tensor): raise NotImplementedError()


# TODO: add regression and point-cloud-to-point-cloud support