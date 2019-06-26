from fastai.core import *
from fastai.torch_core import *
import pyntcloud

__all__ = ['PtCloudItem', 'PtCloudSegmentItem', 'PtCloudUpsampledItem', 'ptcloud_split',
           'ptcloud_sample', 'ptcloud_voxel_sample', 'ptcloud_norm_xyz', 'PtCloudTransform',
           'PtCloudXYZTransform', 'PtCloudAffineTransform', 'PtCloudFeaturesTransform',
           'PtCloudIdxTransform', 'PtCloudMaskOutTransform', 'PtCloudIdxTransformX', 'PtCloudIdxTransformY']


class PtCloudItemBase(ItemBase):
    def __repr__(self): return f'{self.__class__.__name__} {tuple(self.data.shape)}'

    def clone(self): return self.__class__(self.data.clone())

    @property
    def device(self): return self.data.device

    @property
    def n_points(self): return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1] if len(self.data) > 1 else self.data.shape[0]

    @property
    def xyz(self): return self.data[:, 0:3]

    @xyz.setter
    def xyz(self, xyz): self.data[:, 0:3] = xyz

    @property
    def features(self): return self.data[:, 3:]

    @features.setter
    def features(self, features): self.data[:, 3:] = features

    @property
    def shape(self): return self.data.shape

    def refresh(self): return self

    def apply_tfms(self,
                   tfms: Collection,
                   do_resolve: bool = True,
                   xtra: Dict[Callable, dict] = None,
                   ):
        if not (tfms or xtra):
            return self

        tfms = sorted(listify(tfms), key=lambda o: o.tfm.order)
        xtra = ifnone(xtra, {})
        if do_resolve:
            for tfm in tfms:
                tfm.resolve()

        x = self.clone()
        for tfm in tfms:
            x = tfm(x)

        return x.refresh()

    def apply_tfm_xyz(self, func, *args, **kwargs):
        self.xyz = func(self.xyz, *args, **kwargs)
        return self

    def apply_tfm_affine(self, func, *args, **kwargs):
        return self.apply_tfm_xyz(
            lambda xyz: xyz @ tensor(func(*args, **kwargs)).to(self.device))

    def apply_tfm_features(self, func, *args, **kwargs):
        if self.n_features:
            self.features = func(self.features, *args, **kwargs)
        return self

    def apply_tfm_idx(self, func, *args, **kwargs):
        self.data = self.data[func(self.n_points, *args, **kwargs), ...]
        return self

    apply_tfm_idx_x = apply_tfm_idx
    apply_tfm_idx_y = apply_tfm_idx


class PtCloudItem(PtCloudItemBase):

    @classmethod
    def from_ptcloud(cls,
                     ptcloud: pyntcloud.PyntCloud,
                     features: Union[str, Collection[str]] = ('x', 'y', 'z')):
        features = listify(features)
        pts = ptcloud.points[features]
        return cls(torch.from_numpy(np.asarray(pts, dtype=np.float32)))

    def apply_tfm_mask_out(self, func, *args, **kwargs):
        self.data[func(self.n_points, *args, **kwargs), ...] = 0
        return self

    def apply_tfm_idx_y(self, func, *args, **kwargs):
        return self


class PtCloudSegmentItem(PtCloudItemBase):
    @classmethod
    def from_ptcloud(cls,
                     ptcloud: pyntcloud.PyntCloud,
                     label_field: str = 'classification'):
        labels = ptcloud.points[label_field].values
        return cls(torch.from_numpy(labels).long())

    def reconstruct(self, t: Tensor):
        return PtCloudSegmentItem(t)

    def apply_tfm_xyz(self, func, *args, **kwargs):
        return self

    def apply_tfm_features(self, func, *args, **kwargs):
        return self

    def apply_tfm_mask_out(self, func, *args, **kwargs):
        self.data[func(self.n_points, *args, **kwargs), ...] = -1
        return self

    def apply_tfm_idx_x(self, func, *args, **kwargs):
        return self


class PtCloudUpsampledItem(PtCloudItemBase):
    @classmethod
    def from_ptcloud(cls,
                     ptcloud: pyntcloud.PyntCloud,
                     features: Union[str, Collection[str]] = ('x', 'y', 'z')):
        features = listify(features)
        pts = ptcloud.points[features]
        return cls(torch.from_numpy(np.asarray(pts, dtype=np.float32)))

    def reconstruct(self, t: Tensor):
        return PtCloudSegmentItem(t)

    def apply_tfm_idx(self, func, *args, **kwargs):
        return self

    def apply_tfm_mask_out(self, func, *args, **kwargs):
        return self

    def apply_tfm_idx_x(self, func, *args, **kwargs):
        return self


def ptcloud_norm_xyz(pts: pyntcloud.PyntCloud, scale: float = None):
    scale = listify(scale or 1, 3)

    df = pts.points.assign(
        x=(pts.points.x - pts.points.x.mean()) * scale[0],
        y=(pts.points.x - pts.points.x.mean()) * scale[1],
        z=(pts.points.x - pts.points.x.mean()) * scale[2],
    )
    return pyntcloud.PyntCloud(df)


def ptcloud_split(pts: pyntcloud.PyntCloud, cell_size: Union[float, Iterable]
                  ) -> List[pyntcloud.PyntCloud]:
    cell_size = listify(cell_size, 3)

    df = pts.points.assign(
        group_x=pts.points.x // cell_size[0],
        group_y=pts.points.y // cell_size[1],
        group_z=pts.points.z // cell_size[2],
    )
    return [
        pyntcloud.PyntCloud(group.drop(columns=['group_x', 'group_y', 'group_z'])
                            .reset_index(drop=True))
        for _, group in df.groupby(by=['group_x', 'group_y', 'group_z'], as_index=False)
    ]


def ptcloud_sample(pts: pyntcloud.PyntCloud, n: int):
    return pyntcloud.PyntCloud(pts.points.sample(n, replace=True).reset_index(drop=True))


def ptcloud_voxel_sample(pts: pyntcloud.PyntCloud, voxel_size: Union[Iterable, float],
                         agg='intensity', keep_max=True):
    voxel_size = listify(voxel_size, 3)

    df = pts.points.assign(
        group_x=pts.points.x // voxel_size[0],
        group_y=pts.points.y // voxel_size[1],
        group_z=pts.points.z // voxel_size[2],
    )  # type: pd.DataFrame
    df.sort_values(agg, inplace=True)
    df.drop_duplicates(['group_x', 'group_y', 'group_z'],
                       keep='last' if keep_max else 'first')
    return pyntcloud.PyntCloud(df.reset_index(drop=True))


def _get_default_args(func:Callable):
    return {k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty}


class PtCloudTransform:
    "Utility class for adding probability and wrapping support to transform `func`."
    order, _wrap = 0, None
    def __init__(self, func: Callable, order: int = None):
        "Create a transform for `func` and assign it an priority `order`, attach to `PtCloudItem` class."
        self.order = ifnone(order, self.order)
        self.func = func
        self.func.__name__ = func.__name__[1:]  # To remove the _ that begins every transform function.
        functools.update_wrapper(self, self.func)
        self.func.__annotations__['return'] = Tensor
        self.params = copy(func.__annotations__)
        self.def_args = _get_default_args(func)
        setattr(PtCloudItem, func.__name__,
                lambda x, *args, **kwargs: self.calc(x, *args, **kwargs))

    def __call__(self, *args: Any, p: float = 1., is_random: bool = True,
                 use_on_y: bool = True, **kwargs):
        "Calc now if `args` passed; else create a transform called prob `p` if `random`."
        if args: return self.calc(*args, **kwargs)
        else: return PtCloudRandTransform(self, kwargs=kwargs, is_random=is_random,
                                   use_on_y=use_on_y, p=p)

    def calc(self, x: PtCloudItem, *args: Any, **kwargs: Any) -> PtCloudItem:
        "Apply point cloud to `x`, wrapping if necesssary"
        if self._wrap: return getattr(x, self._wrap)(self.func, *args, **kwargs)
        else: return self.func(x, *args, **kwargs)

    @property
    def name(self) -> str: return self.__class__.__name__

    def __repr__(self) -> str: return f'{self.name} ({self.func.__name__})'


@dataclass
class PtCloudRandTransform:
    "Wrap `Transform` to add randomized execution."
    tfm: PtCloudTransform
    kwargs: dict
    p: float = 1.0
    resolved: dict = field(default_factory=dict)
    do_run: bool = True
    is_random: bool = True
    use_on_y: bool = True

    def __post_init__(self):
        functools.update_wrapper(self, self.tfm)

    def resolve(self) -> None:
        "Bind any random variables in the transform."
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # for each param passed to tfm...
        for k, v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else:
                self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k, v in self.tfm.def_args.items():
            if k not in self.resolved: self.resolved[k] = v
        # anything left over must be callable without params
        for k, v in self.tfm.params.items():
            if k not in self.resolved and k != 'return': self.resolved[k] = v()

        self.do_run = rand_bool(self.p)

    @property
    def order(self) -> int:
        return self.tfm.order

    def __call__(self, x: PtCloudItem, *args, **kwargs) -> PtCloudItem:
        "Randomly execute our tfm on `x`."
        return self.tfm(x, *args, **{**self.resolved, **kwargs}) if self.do_run else x


class PtCloudXYZTransform(PtCloudTransform):
    _wrap = 'apply_tfm_xyz'


class PtCloudAffineTransform(PtCloudTransform):
    _wrap = 'apply_tfm_affine'


class PtCloudFeaturesTransform(PtCloudTransform):
    _wrap = 'apply_tfm_features'


class PtCloudIdxTransform(PtCloudTransform):
    _wrap = 'apply_tfm_idx'


class PtCloudMaskOutTransform(PtCloudTransform):
    _wrap = 'apply_tfm_mask_out'


class PtCloudIdxTransformX(PtCloudTransform):
    _wrap = 'apply_tfm_idx_x'


class PtCloudIdxTransformY(PtCloudTransform):
    _wrap = 'apply_tfm_idx_y'
