import cv2
import numpy as np
import torch
import torch.nn as nn

from network import EncDecNet
from rightLaneData import getRightLaneDatasets, getDataLoaders
from trainer import trainNet

haveCuda = torch.cuda.is_available()


def trainEncDecNet(nFeat, nLevels, kernelSize=3, nLinType='relu', bNorm=True,
                   dropOut=0.3, bSize=32, lr=1e-3, lr_ratio=1000, numEpoch=50, decay=1e-4,
                   verbose=False, setSeeds=True):
    # A függvény ismételt futtatása esetén ugyanazokat az eredményeket adja
    if setSeeds:
        torch.manual_seed(42)
        if haveCuda:
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Létrehozzuk a hálózatunkat (lehetőség szerint GPU-n); az osztályok száma adott az adatbázis miatt!
    net = EncDecNet(nFeat, nLevels, kernelSize, nLinType, bNorm, dropOut)
    if haveCuda:
        net = net.cuda()

    datasets = getRightLaneDatasets('./data')

    # Adatbetöltők lekérése adott batch mérettel
    dataloaders = getDataLoaders(datasets, bSize)

    bestValAcc = trainNet(net=net, dataLoaders=dataloaders, lr=lr, lr_ratio=lr_ratio, numEpoch=numEpoch, decay=decay,
                          verbose=verbose, setSeeds=False)

    return bestValAcc, net


def makeExamples(net: nn.Module, testLoader, printNum):
    assert printNum >= 2

    net = net.cpu()
    x, y = next(iter(testLoader))
    x = x[:printNum]
    y = y[:printNum]

    _, p = torch.max(net(x), 1)
    p = p.squeeze()

    x = (x * 255).byte()
    y = (y * 255).byte()
    p = (p * 255).byte()

    finalResult = np.empty([0, 480, 3], dtype=np.uint8)
    for img, label, pred in zip(x, y, p):
        img = np.transpose(img.numpy(), axes=(1, 2, 0))
        label = label.numpy()
        pred = pred.numpy()

        label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

        result = np.concatenate((img, pred, label), axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('./results/preds.png', finalResult)


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    import logging

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    bestAcc, net = trainEncDecNet(16, 4, 5, 'leakyRelu', bSize=512, verbose=True, numEpoch=100)
    print(f"A teszt adatokon elért pontosság: {bestAcc:.2f}%")
    torch.save(net.state_dict(), './results/EncDecNet.pth')

    # Print some example predictions
    net.load_state_dict(torch.load('./results/EncDecNet.pth'))
    net.eval()
    datasets = getRightLaneDatasets('./data')
    _, _, trainLoader = getDataLoaders(datasets, 8)
    makeExamples(net, trainLoader, 5)
