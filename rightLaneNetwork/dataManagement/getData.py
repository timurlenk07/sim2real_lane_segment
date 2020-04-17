import concurrent.futures
import os

import torch

from .RightLaneDataset import RightLaneDataset
from .databasePreprocessing import video2images


def getRightLaneDatasets(dataPath, transform=None, shouldPreprocess=False, preprocessTransform=None):
    # assert dataPath is a valid path
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    train_dir = os.path.join(dataPath, "train")
    valid_dir = os.path.join(dataPath, "validation")
    test_dir = os.path.join(dataPath, "test")
    dataPaths = [train_dir, valid_dir, test_dir]

    for i, directory in enumerate(dataPaths):
        # Assert directory already exist (else there would be no data)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist!")

        # In case no input or target exist, try to do preprocessing
        input_dir = os.path.join(dataPath, 'input')
        target_dir = os.path.join(dataPath, 'label')
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            shouldPreprocess = True

    if shouldPreprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(video2images, dataPath, preprocessTransform, True, False)
                       for dataPath in dataPaths]
            concurrent.futures.as_completed(futures)

    datasets = (RightLaneDataset(dataPath, transform, True) for dataPath in dataPaths)

    return datasets


def getRightLaneDatasetsMME(dataPath, transform=None, shouldPreprocess=False, preprocessTransform=None):
    # assert dataPath is a valid path
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    # Check if data directories are available
    source_path = os.path.join(dataPath, "source")
    target_path = os.path.join(dataPath, 'target')
    dataPaths = [source_path, target_path]
    for directory in dataPaths:
        # Assert directories already exist (else there would be no data)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist, hence no data is available!")

    target_train_path = os.path.join(target_path, 'train')
    target_unlabelled_path = os.path.join(target_path, 'unlabelled')
    target_test_path = os.path.join(target_path, 'test')
    dataPaths = [source_path, target_train_path, target_unlabelled_path, target_test_path]
    labelled = [True, True, False, True]

    for i, directory in enumerate(dataPaths):
        # Assert directory already exist (else there would be no data)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist!")

        # In case no input exist, try to do preprocessing (labels might not exist)
        input_dir = os.path.join(dataPath, 'input')
        if not os.path.exists(input_dir):
            shouldPreprocess = True

    if shouldPreprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(video2images, dataPath, preprocessTransform, labels, False)
                       for dataPath, labels in zip(dataPaths, labelled)]
            concurrent.futures.as_completed(futures)

    datasets = (RightLaneDataset(dataPath, transform, labels) for dataPath, labels in zip(dataPaths, labelled))

    return datasets


def getDataLoaders(datasets, batchSize=128):
    trainLoader = torch.utils.data.DataLoader(datasets[0], batch_size=batchSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(datasets[1], batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(datasets[2], batch_size=batchSize, shuffle=True)
    return trainLoader, validLoader, testLoader


def getDataLoadersMME(datasets, batchSize=128):
    sourceLoader = torch.utils.data.DataLoader(datasets[0], batch_size=batchSize, shuffle=True)
    targetTrainLoader = torch.utils.data.DataLoader(datasets[1], batch_size=batchSize, shuffle=True)
    targetUnlabelledLoader = torch.utils.data.DataLoader(datasets[2], batch_size=batchSize, shuffle=True)
    targetTestLoader = torch.utils.data.DataLoader(datasets[3], batch_size=batchSize, shuffle=True)
    return sourceLoader, targetTrainLoader, targetUnlabelledLoader, targetTestLoader
