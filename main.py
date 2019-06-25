import sys
import threading


def setup_thread_excepthook():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540

    Call once from the main thread before creating any threads.
    """

    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):

        init_original(self, *args, **kwargs)
        run_original = self.run

        def run_with_except_hook(*args2, **kwargs2):
            try:
                run_original(*args2, **kwargs2)
            except Exception:
                sys.excepthook(*sys.exc_info())

        self.run = run_with_except_hook

    threading.Thread.__init__ = init

if __name__ == "__main__":
    setup_thread_excepthook()

    from pointcloud import *
    from fastai.vision

    data = (PtCloudUpsampleList
            .from_folder('test-data')
            .chunkify(10)
            .split_by_rand_pct()
            .label(downsample_cellsize=0.2, downsample_agg='x')
            .transform(get_transforms(), tfm_y=True)
            .databunch(bs=2)
            .normalize()
            )

<<<<<<< HEAD
    print(data)

    labellist = splitlist.label_from_fields('x', classes=['None', 'Marking'])
    print(labellist)

    databunch = labellist.databunch(bs=1).normalize()
    print(databunch)
=======
    learn = pcn_learner(data, pointnet2_msg_seg)
    print(learn)
>>>>>>> b74d0e3d549047cadb8cd72d61cbe2a43b479b53
