import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from rightLaneData import getRightLaneDatasets, getDataLoaders

haveCuda = torch.cuda.is_available()


def makeEpoch(net, dataLoader, isTrain, criterion, optimizer=None, useCuda=False, verbose=False):
    if isTrain and optimizer is None:
        raise ValueError("When training optimizer cannot be None!")

    # Mutatók számításához változók
    runningLoss = 0.0
    correct = 0
    total = 0

    # A hálót train vagy eval módba állítjuk (batch normalizationt és dropoutot befolyásolja)
    if isTrain:
        net.train()
    else:
        net.eval()

    # Iteráció az epochban, batchenként
    for data in (tqdm(dataLoader) if verbose else dataLoader):
        # Betöltött adat feldolgozása, ha kell, GPU-ra töltés
        inputs, labels = data
        if useCuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Tanítás  esetén a gradiensek törlése
        if isTrain:
            optimizer.zero_grad()

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Tanítás esetén gradiensek számolása, optimizálóval lépés
        if isTrain:
            loss.backward()
            optimizer.step()

        # Mutatók frissítése: költség összegzés
        runningLoss += loss.item()

        # Predikció számítás (Maximum Likelihood elv alapján)
        _, predicted = torch.max(outputs, 1)

        # Helyes és összes predikció frissítése
        total += labels.numel()
        correct += predicted.eq(labels).sum().item()

    # Költség és pontosság meghatározása az epoch mutatóiból
    epochLoss = runningLoss / len(dataLoader)
    epochCorr = correct / total * 100

    return epochLoss, epochCorr


def trainNet(net, dataLoaders, lr=1e-3, lr_ratio=1000, numEpoch=50, decay=1e-4, verbose=False, setSeeds=True):
    if numEpoch < 1:
        return 0.0

    # A függvény ismételt futtatása esetén ugyanazokat az eredményeket adja
    if setSeeds:
        torch.manual_seed(42)
        if haveCuda:
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Adatbetöltők kicsomagolása
    trainLoader, validLoader, testLoader = dataLoaders

    # Költség és optimalizáló algoritmus (ezek előre meghatározottak, paramétereik szabadok)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=decay)

    # Tanulási ráta ütemező: az ütemezés frekvenciája az epochok számának ötöde, de minimum 5
    scheduler = CosineAnnealingLR(optimizer, max(numEpoch // 5, 5), eta_min=lr / lr_ratio)

    # A futás során keletkező mutatók mentésére üres listák
    trLosses = []
    trAccs = []
    valLosses = []
    valAccs = []

    if verbose:
        logging.info(f"Training start. Number of parameters: {net.getNParams()}")

    # Save best model
    maxAcc = 0

    # Iteráció az epochokon
    for epoch in range(numEpoch):

        # Tanítás és validáció
        trLoss, trAcc = makeEpoch(net, trainLoader, True, criterion, optimizer, useCuda=haveCuda, verbose=verbose)
        valLoss, valAcc = makeEpoch(net, validLoader, False, criterion, useCuda=haveCuda, verbose=verbose)

        # Mutatók mentése
        trLosses.append(trLoss)
        trAccs.append(trAcc)
        valLosses.append(valLoss)
        valAccs.append(valAcc)

        # Tanításról információ kiírása
        if verbose:
            logging.info(f"Epoch {epoch + 1}: Training cost is {trLoss:.3f}, train accuracy is {trAcc:.2f}%")
            logging.info(f"Epoch {epoch + 1}: Validation cost is {valLoss:.3f}, validation accuracy is {valAcc:.2f}%")

        # Tanulási ráta ütemező léptetése
        scheduler.step()

        if (valAcc > maxAcc):
            maxAcc = valAcc
            torch.save(net.state_dict(), '.bestModel')

    net.load_state_dict(torch.load('.bestModel'))
    os.remove('.bestModel')

    # Calculate result for test set
    testLoss, testAcc = makeEpoch(net, testLoader, False, criterion, useCuda=haveCuda, verbose=verbose)

    # Befejezéskor információk kiírása
    if verbose:
        logging.info('Training complete.')
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(trLosses, label='Training')
        ax1.plot(valLosses, label='Validation')
        ax1.set(xlabel='Epoch', ylabel='Loss')
        ax1.legend()

        ax2.plot(trAccs, label='Training')
        ax2.plot(valAccs, label='Validation')
        ax2.set(xlabel='Epoch', ylabel='Accuracy [%]')
        ax2.legend()

        plt.tight_layout()
        fig.savefig('./results/trainingChart.png')
        plt.close()

    # Visszatérési érték a tanítás jóságát jellemző érték: teszt pontosság
    return testAcc


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    datasets = getRightLaneDatasets('./data', (160, 120))
    dataloaders = getDataLoaders(datasets, 8)
    bestAcc, net = trainNet(verbose=True, numEpoch=1)
    print(f"Teszt pontosság a próbatanítás során: {bestAcc:.2f}%")

    # Evaluate on test
    net.eval()
