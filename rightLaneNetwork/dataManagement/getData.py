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
        input_dir = os.path.join(dataPath, 'orig')
        target_dir = os.path.join(dataPath, 'annot')
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            shouldPreprocess = True

    if shouldPreprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(video2images, dataPath, preprocessTransform, False) for dataPath in dataPaths]
            concurrent.futures.as_completed(futures)

    datasets = (RightLaneDataset(dataPath, transform) for dataPath in dataPaths)

    return datasets


def getDataLoaders(datasets, batchSize=128):
    trainLoader = torch.utils.data.DataLoader(datasets[0], batch_size=batchSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(datasets[1], batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(datasets[2], batch_size=batchSize, shuffle=True)
    return trainLoader, validLoader, testLoader
