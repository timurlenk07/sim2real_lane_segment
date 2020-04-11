import math

import torch
from bayes_opt import BayesianOptimization

from models.EncDecNet import activationTypes
from old.rightLaneSegment import trainEncDecNet

assert torch.cuda.device_count() <= 1


def trainEncDecNetHyperopt(nFeat_exp, nLevels, kernelSize, nLinType, dropOut, bSize_exp,
                           lr_exp, lr_ratio_exp, decay_exp, grayscale):
    # A hiperparaméter optimalizációhoz szükséges a nem megfelelő paraméterek konverziója
    # A paramétereknek azonos tartományban érdemes lenniük

    # Exponens -> integer -> számítás
    nFeat_exp, bSize_exp = int(round(nFeat_exp)), int(round(bSize_exp))

    nFeat = 2 ** nFeat_exp
    bSize, lr = 2 ** bSize_exp, 10 ** lr_exp
    lr_ratio, decay = 10 ** lr_ratio_exp, 10 ** decay_exp

    # Egyéb integer változók: float -> int
    nLevels, kernelSize = int(round(nLevels)), int(round(kernelSize))

    # Kernel méret: legyen páratlan egész szám!
    kernelSize = kernelSize + (kernelSize % 2) - 1

    # Aktiváció típusa: egész szám, kiválasztja a listából a kapcsolódó sztringet
    nLinType = int(round(nLinType))
    nLinType = activationTypes[nLinType]

    grayscale = int(round(grayscale))
    if grayscale == 0:
        grayscale = False
    else:
        grayscale = True

    acc, net = trainEncDecNet(nFeat, nLevels, kernelSize, nLinType, bNorm=True,
                              dropOut=dropOut, grayscale=grayscale, bSize=bSize, lr=lr, lr_ratio=lr_ratio,
                              numEpoch=50, decay=decay, verbose=False, setSeeds=True)

    return acc - math.log(net.getNParams(), 10) * 5


# A paraméterek határait/lehetséges értékeit definiáljuk
parameterBounds = {'nFeat_exp': (2, 6),
                   'nLevels': (2, 4),
                   'kernelSize': (3, 7),
                   'nLinType': (0, len(activationTypes) - 1),
                   'dropOut': (0, 0.85),
                   'bSize_exp': (3, 9),
                   'lr_exp': (-4, 0),
                   'lr_ratio_exp': (1, 4),
                   'decay_exp': (-6, -2),
                   'grayscale': (0, 1)}

# Optimizáló objektum létrehozása
optimizer = BayesianOptimization(
    f=trainEncDecNetHyperopt,
    pbounds=parameterBounds,
    random_state=42,
)

# Mérjük az optimizáció idejét
import time
import datetime

# Optimalizáció
start = time.time()
optimizer.maximize(
    init_points=5,
    n_iter=45
)
end = time.time()

duration = end - start
print(f"Running time: {datetime.timedelta(seconds=int(round(duration)))} [h:m:s].")

print(f"Best value obtained with hyper-opt: {optimizer.max['target']:.2f}")
print(f"Best settings: {optimizer.max['params']}")
