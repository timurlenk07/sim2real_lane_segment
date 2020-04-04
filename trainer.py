from tqdm import tqdm
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt

from network import EncDecNet
from dataLoaders import getRightLaneDatasets

haveCuda = torch.cuda.is_available()

trainSet, validSet, testSet = getRightLaneDatasets('./data')

# Adatbetöltőket készítő függvény
def getDataLoaders(batchSize=128):
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True)
    return trainLoader, validLoader, testLoader

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
    for data in tqdm(dataLoader):
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
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Költség és pontosság meghatározása az epoch mutatóiból
    epochLoss = runningLoss / len(dataLoader)
    epochCorr = correct / total * 100

    return epochLoss, epochCorr



def trainNet(nFeat, nLevels, kernelSize=3, nLinType='relu', bNorm=True,
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

    # Adatbetöltők lekérése adott batch mérettel
    trainLoader, validLoader, testLoader = getDataLoaders(bSize)

    # Költség és optimalizáló algoritmus (ezek előre meghatározottak, paramétereik szabadok)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=decay)

    # Tanulási ráta ütemező: az ütemezés frekvenciája az epochok számának ötöde, de minimum 10
    scheduler = CosineAnnealingLR(optimizer, max(numEpoch / 5, 10), eta_min=lr / lr_ratio)

    # A futás során keletkező mutatók mentésére üres listák
    trLosses = []
    trAccs = []
    valLosses = []
    valAccs = []

    if verbose:
        print(f"Tanítás indul. Paraméterek száma: {net.getNParams()}")

    # Iteráció az epochokon
    for epoch in range(numEpoch):

        # Tanítás és validáció
        trLoss, trCorr = makeEpoch(net, trainLoader, True, criterion, optimizer, useCuda=haveCuda, verbose=verbose)
        valLoss, valCorr = makeEpoch(net, validLoader, False, criterion, useCuda=haveCuda, verbose=verbose)

        # Mutatók mentése
        trLosses.append(trLoss)
        trAccs.append(trCorr)
        valLosses.append(valLoss)
        valAccs.append(valCorr)

        # Tanításról információ kiírása
        if verbose:
            print(f"Epoch {epoch + 1}: A tanítási költség {trLoss:.3f}, a tanítási pontosság {trCorr:.2f}%")
            print(f"Epoch {epoch + 1}: A validciós költség {valLoss:.3f}, a validációs pontosság {valCorr:.2f}%")

        # Tanulási ráta ütemező léptetése
        scheduler.step()

    # Befejezéskor információk kiírása
    if verbose:
        print('Tanítás befejezve.')
        plt.plot(trLosses, label='Training')
        plt.plot(valLosses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('losses.png', bbox_inches='tight')
        plt.close()

        plt.plot(trAccs, label='Training')
        plt.plot(valAccs, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.legend()
        plt.savefig('accs.png', bbox_inches='tight')
        plt.close()

    # Visszatérési érték a tanítás jóságát jellemző érték: legalacsonyabb validációs költség
    return max(valAccs)



if __name__=='__main__':
    bestAcc = trainNet(64, 3, 7, 'leakyRelu', dropOut=0.5, bSize=1, lr=1e-3, numEpoch=10, verbose=True)
    print(f"Legjobb validációs pontosság a próbatanítás során: {bestAcc}%")