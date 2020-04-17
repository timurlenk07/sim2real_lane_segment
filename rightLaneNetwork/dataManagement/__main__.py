import logging

import torch

from .basicTransforms import SavedTransform, LoadedTransform
from .getData import getRightLaneDatasets

if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    # Adatbázis építés
    datasets = getRightLaneDatasets('./data',
                                    transform=LoadedTransform(grayscale=False, newRes=(160, 120)),
                                    shouldPreprocess=True,
                                    preprocessTransform=SavedTransform(grayscale=True, newRes=(320, 240)),
                                    )
    trainSet, validSet, testSet = datasets
    print(f"Dataset lengths: {len(trainSet)}, {len(validSet)}, {len(testSet)}")

    x, y = trainSet[0]
    print(f"Dataset returned data with input: {x.shape}, and output: {y.shape}")

    # Lekérjük, hogy rendelkezésre áll-e GPU
    haveCuda = torch.cuda.is_available()

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))
