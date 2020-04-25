import logging

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from .getData import getRightLaneDatasets, getRightLaneDatasetsMME
from .myDatasets import ParallelDataset

assert torch.cuda.device_count() <= 1
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
haveCuda = torch.cuda.is_available()


def myTransformation(img, label):
    newRes = (120, 160)
    img = TF.to_pil_image(img)
    img = TF.to_grayscale(img)
    img = TF.resize(img, newRes, interpolation=Image.LANCZOS)
    img = TF.to_tensor(img)

    if label is not None and len(label.shape) >= 2:
        label = TF.to_pil_image(label)
        label = TF.resize(label, newRes, interpolation=Image.LANCZOS)
        label = TF.to_tensor(label).squeeze()
        label[label > 0.5] = 1
        label[label != 1] = 0
        label = label.long()

    return img, label


def makeTests():
    # Adatbázis építés
    datasets = getRightLaneDatasets('./data', transform=myTransformation)
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
    datasets = getRightLaneDatasetsMME('./data', transform=myTransformation)
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

    pd = ParallelDataset(sourceSet, targetUnlabelledSet)
    dl = DataLoader(pd, 4, False)
    batch = next(iter(dl))
    print(type(batch))
    print(len(batch))
    a, b, c, d = batch
    print(a.shape, b.shape, c.shape, d.shape)


makeTestsMME()
