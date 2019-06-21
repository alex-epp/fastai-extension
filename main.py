if __name__ == "__main__":
    from pointcloud import *
    from fastai.basic_train import *

    item_list = PtCloudList.from_folder('test-data')
    print(item_list)

    laern = Learner()