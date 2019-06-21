if __name__ == "__main__":
    from pointcloud import *
    from fastai.vision import data

    item_list = PtCloudList.from_folder('test-data')
    print(item_list)
