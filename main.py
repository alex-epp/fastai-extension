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

    labellist = splitlist.label_from_fields('x', classes=['None', 'Marking'])
    print(labellist)

    databunch = labellist.databunch(bs=1).normalize()
    print(databunch)
