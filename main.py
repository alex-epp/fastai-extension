import numpy as np
import contextlib
import time
from pyntcloud import PyntCloud
from fastai.vision import *

from util.timing import process_timer


if __name__ == "__main__":
    with process_timer('Loading'):
        cloud: PyntCloud = PyntCloud.from_file('test-data/fragment.ply')

    with process_timer('Finding kneighbours'):
        ev = cloud.add_scalar_field("eigen_values", k_neighbors=cloud.get_neighbors(k=45))

    with process_timer('Finding anistropy'):
        cloud.add_scalar_field('anisotropy', ev=ev)

    with process_timer('Saving'):
        cloud.to_file('test-data/fragment-txt.ply', as_text=True)
