import logging

import torch

from .basicTransforms import SavedTransform, LoadedTransform
from .getData import getRightLaneDatasets, getRightLaneDatasetsMME

assert torch.cuda.device_count() <= 1
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
haveCuda = torch.cuda.is_available()


def makeTests():
    # Adatbázis építés
    datasets = getRightLaneDatasets('./data',
                                    transform=LoadedTransform(grayscale=False, newRes=(160, 120)),
                                    shouldPreprocess=True,
                                    preprocessTransform=SavedTransform(grayscale=True, newRes=(320, 240)),
                                    )
    trainSet, validSet, testSet = datasets
    print(f"Dataset lengths: {len(trainSet)}, {len(validSet)}, {len(testSet)}")

    x, y = trainSet[0]
    print(f"trainSet returned data with input: {x.shape}, and output: {y.shape}")

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))


def makeTestsMME():
    # Adatbázis építés
    datasets = getRightLaneDatasetsMME('./data',
                                       transform=LoadedTransform(grayscale=False, newRes=(160, 120)),
                                       shouldPreprocess=True,
                                       preprocessTransform=SavedTransform(grayscale=True, newRes=(320, 240)),
                                       )
    sourceSet, targetTrainSet, targetUnlabelledSet, targetTestSet = datasets
    print(f"Dataset lengths: {len(sourceSet)}, {len(targetTrainSet)}, " +
          f"{len(targetUnlabelledSet)}, {len(targetTestSet)}")

    x, y = sourceSet[0]
    print(f"sourceSet returned data with input: {x.shape}, and output: {y.shape}")

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))


makeTestsMME()
