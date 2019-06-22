if __name__ == "__main__":
    from pointcloud import *

    data = (PtCloudSegmentationList
            .from_folder('test-data')
            .voxel_sample(0.2, agg='x')
            .chunkify(1)
            .split_by_rand_pct()
            .label_from_field('x', classes=['None', 'Marking'])
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=1)
            .normalize()
            )

    print(data)
