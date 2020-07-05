import logging
import os

import torch
from torch.utils.data import DataLoader

from .myDatasets import ParallelDataset, RightLaneDataset
from .myTransforms import testTransform

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
haveCuda = torch.cuda.is_available()


def makeTests():
    # Database building
    trainSet = RightLaneDataset(os.path.join('data', 'train'), testTransform, haveLabels=True)
    validSet = RightLaneDataset(os.path.join('data', 'valid'), testTransform, haveLabels=True)
    testSet = RightLaneDataset(os.path.join('data', 'test'), testTransform, haveLabels=True)
    print(f"Dataset lengths: {len(trainSet)}, {len(validSet)}, {len(testSet)}")

    x, y = trainSet[0]
    print(f"trainSet returned data with input: {x.shape}, and output: {y.shape}")

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))


def makeTestsMME():
    # Database building
    sourceSet = RightLaneDataset(os.path.join('data', 'source'), testTransform, haveLabels=True)
    targetTrainSet = RightLaneDataset(os.path.join('data', 'target', 'train'), testTransform, haveLabels=True)
    targetUnlabelledSet = RightLaneDataset(os.path.join('data', 'target', 'unlabelled'),
                                           testTransform, haveLabels=False)
    targetTestSet = RightLaneDataset(os.path.join('data', 'target', 'test'), testTransform, haveLabels=True)
    print(f"Dataset lengths: {len(sourceSet)}, {len(targetTrainSet)}, " +
          f"{len(targetUnlabelledSet)}, {len(targetTestSet)}")

    x, y = sourceSet[0]
    print(f"sourceSet returned data with input: {x.shape}, and output: {y.shape}")

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))

    pd = ParallelDataset(sourceSet, targetUnlabelledSet)
    dl = DataLoader(pd, 4, False)
    batch = next(iter(dl))
    print(type(batch))
    print(len(batch))
    a, b, c, d = batch
    print(a.shape, b.shape, c.shape, d.shape)

# Uncomment the tests to perform them
# makeTests()
# makeTestsMME()
