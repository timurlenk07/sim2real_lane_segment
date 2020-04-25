import os

from .myDatasets import RightLaneDataset


def getRightLaneDatasets(dataPath, transform=None):
    train_dir = os.path.join(dataPath, "train")
    valid_dir = os.path.join(dataPath, "validation")
    test_dir = os.path.join(dataPath, "test")
    dataPaths = [train_dir, valid_dir, test_dir]

    return tuple(RightLaneDataset(dataPath, transform, True) for dataPath in dataPaths)


def getRightLaneDatasetsMME(dataPath, transform=None):
    source_path = os.path.join(dataPath, "source")
    target_path = os.path.join(dataPath, 'target')

    target_train_path = os.path.join(target_path, 'train')
    target_unlabelled_path = os.path.join(target_path, 'unlabelled')
    target_test_path = os.path.join(target_path, 'test')
    dataPaths = [source_path, target_train_path, target_unlabelled_path, target_test_path]
    labelled = [True, True, False, True]

    return tuple(RightLaneDataset(dataPath, transform, haveLabels) for dataPath, haveLabels in zip(dataPaths, labelled))
