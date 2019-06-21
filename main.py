if __name__ == "__main__":
    from pointcloud import *
    from fastai.vision import data

    data = (PtCloudList.from_folder('test-data')
            .chunkify(1)
            .norm_xyz()
            .random_sample(1024)
            .split_by_rand_pct()
            .label_from_field('x')
            .databunch(bs=3)
            )

