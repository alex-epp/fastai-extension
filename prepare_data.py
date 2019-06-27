import numpy as np
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
from warnings import warn

from pyntcloud import PyntCloud
from .pointcloud import ptcloud_split, ptcloud_voxel_sample


def extract(source_path: Path, target_path: Path, n_files=10):
    if not source_path.exists():
        warn(f'Source path {source_path} doesn\'t seem to exist. Skipping unzip step')
        return

    for file in tqdm(sorted(source_path.glob('*.zip'))[:n_files]):
        unzipped_path = target_path / file.stem
        if not unzipped_path.is_file():
            with ZipFile(file, 'r') as zipfile:
                zipfile.extractall(unzipped_path.parent)


def split(in_path: Path, out_path: Path, features=None, cell_size=20):
    if not in_path.exists():
        warn(f'Input path {in_path} doesn\'t seem to exist. Skipping split step')
        return

    features = ['x', 'y', 'z'] + (features or [])

    for file in tqdm(sorted(in_path.glob('*.las'))):
        split_path = out_path / file.stem
        if not split_path.is_dir():
            for i, p in enumerate(ptcloud_split(PyntCloud.from_file(file), cell_size)):
                p.points[features].to_pickle(split_path / f'{i}.pkl')


def downsample(in_path: Path, out_path: Path, voxel_size=0.2):
    if not in_path.exists():
        warn(f'Input path {in_path} doesn\'t seem to exist. Skipping split step')
        return

    for file in tqdm(sorted(in_path.glob('**/*.npy'))):
        ds_file = out_path / file.stem
        if not ds_file.exists():
            pts = ptcloud_voxel_sample(PyntCloud.from_file(file), voxel_size)
            pts.points.to_pickle(ds_file)


if __name__ == '__main__':
    features = ['intensity']

    zip_path = Path('/media/aepp2/MyPassport_Blue/LiDAR_LAS-2013/LiDAR_LAS')
    dataset_path = Path('/media/aepp2/New Volume/pcn')

    extract(zip_path, dataset_path/'orig')

    split(dataset_path/'orig', dataset_path/'chunks', features=features)
    downsample(dataset_path/'chunks', dataset_path/'chunks_ds')

