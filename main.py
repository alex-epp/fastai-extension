if __name__ == "__main__":
    from pointcloud import *

    data = (PtCloudUpsampleList
            .from_folder('test-data')
            .chunkify(10)
            .split_by_rand_pct()
            .label(downsample_cellsize=0.2, downsample_agg='x')
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=2)
            .normalize()
            )

    learn = pcn_learner(data, pointnet2_msg_seg)
    print(learn)
