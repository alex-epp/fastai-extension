from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
from warnings import warn


def extract(source_path: Path, target_path: Path):
    if not source_path.exists():
        warn(f'Source path {source_path} doesn\'t seem to exist. Skipping unzip step')
        return

    n_files = 10

    for file in tqdm(sorted(source_path.glob('*.zip'))[:n_files]):
        unzipped_path = target_path / file.stem
        if not unzipped_path.is_file():
            with ZipFile(file, 'r') as zipfile:
                zipfile.extractall(unzipped_path.parent)


if __name__ == '__main__':
    zip_path = Path('/media/aepp2/MyPassport_Blue/LiDAR_LAS-2013/LiDAR_LAS')
    dataset_path = Path('/media/aepp2/New Volume/pcn')

    extract(zip_path, dataset_path)
