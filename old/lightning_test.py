from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.EncDecNet import Conv
from rightLaneData import getRightLaneDatasets, LoadedTransform, SavedTransform

activationDict = nn.ModuleDict({
    'relu': nn.ReLU(),
    'prelu': nn.PReLU(),
    'leakyRelu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'none': nn.Identity(),
})
activationTypes = list(activationDict.keys())



class EncDecNetLightning(pl.LightningModule):
    def __init__(self, nFeat: int, nLevels: int, kernelSize: int = 3,
                 nLinType: str = 'relu', bNorm: bool = True, dropOut: int = 0.3, inFeat=3,
                 batchSize=64):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.batchSize = batchSize
        self.dataSets = None
        self.lr = 1e-3
        self.decay = 1e-4

        self.__buildModel(nFeat=nFeat, nLevels=nLevels, kernelSize=kernelSize,
                          nLinType=nLinType, bNorm=bNorm, dropOut=dropOut, inFeat=inFeat)

    def __buildModel(self, nFeat: int, nLevels: int, kernelSize: int = 3,
                     nLinType: str = 'relu', bNorm: bool = True, dropOut: int = 0.3, inFeat=3):

        # Kapott paraméterek ellenőrzése
        if nFeat < 1:
            raise ValueError(f"A csatornák számának legalább 1-nek kell lennie! Kapott érték: {nFeat}")
        if nLevels < 1:
            raise ValueError(f"A szintek számának legalább 1-nek kell lennie! Kapott érték: {nLevels}")
        if nLinType not in activationTypes:
            raise ValueError(f"A kapott aktiváció: {nLinType} nincs megvalósítva!\
                                        Válasszon egyet a következők közül: {activationTypes}")

        # Aktivációs függvény a paraméter sztringből
        self.activation = activationDict[nLinType]
        self.pool = nn.MaxPool2d(kernelSize, stride=2, padding=kernelSize // 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Az első szint bemenete 3 csatornás (RGB kép)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Első szint csatornáinak számolása
        oFeat = nFeat

        # Minden enkóder hozzáadása
        for _ in range(nLevels - 1):
            layer = Conv(inFeat, oFeat, kernelSize=kernelSize, activation=self.activation, bNorm=bNorm, dropOut=dropOut)
            self.encoders.append(layer)
            inFeat = oFeat
            oFeat = 2 * oFeat

        # Csatornák újraszámolása
        oFeat = oFeat // 2

        # Minden enkóder hozzáadása
        for _ in range(nLevels - 1):
            layer = Conv(inFeat, oFeat, kernelSize=kernelSize, activation=self.activation, bNorm=bNorm, dropOut=dropOut)
            self.decoders.append(layer)
            inFeat = oFeat
            oFeat = oFeat // 2

        # Pixelenként következtetés, Sigmoid aktiváció
        self.classifier = Conv(inFeat, 2, kernelSize=1, activation=nn.Softmax(dim=-3), bNorm=False, dropOut=0)

    def forward(self, x):
        # A bemenetet végigvezetjük az összes szinten
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        for decoder in self.decoders:
            x = decoder(x)
            x = self.upsample(x)

        # Klasszifikáció
        x = self.classifier(x)

        return x

    def prepare_data(self):
        self.dataSets = getRightLaneDatasets('./data',
                                             transform=LoadedTransform(grayscale=True, newRes=(160, 120)),
                                             shouldPreprocess=True,
                                             preprocessTransform=SavedTransform(grayscale=True, newRes=(160, 120)),
                                             )

    def train_dataloader(self):
        return DataLoader(self.dataSets[0], batch_size=self.batchSize, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataSets[1], batch_size=self.batchSize, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dataSets[2], batch_size=self.batchSize, shuffle=False)

    def configure_optimizers(self):
        optimizer = Adam(net.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 20, eta_min=self.lr / 1000)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # acc
        _, labels_hat = torch.max(outputs, 1)
        train_acc = labels_hat.eq(y).sum() * 100.0 / y.numel()

        progress_bar = {
            'train_acc': train_acc
        }
        logs = {
            'tloss': loss,
            'train_acc': train_acc
        }

        output = OrderedDict({
            'loss': loss,
            'progress_bar': progress_bar,
            'log': logs
        })
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # acc
        _, labels_hat = torch.max(outputs, 1)
        correct = labels_hat.eq(y).sum()
        total = torch.tensor(y.numel())

        output = OrderedDict({
            'val_loss': loss,
            'correct': correct,
            'total': total,
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        total = torch.stack([x['total'] for x in outputs]).sum()
        val_acc = correct * 100.0 / total

        tensorboard_logs = {'val_loss': val_loss,
                            'val_acc': val_acc}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # acc
        _, labels_hat = torch.max(outputs, 1)
        correct = labels_hat.eq(y).sum()
        total = torch.tensor(y.numel())

        output = OrderedDict({
            'test_loss': loss,
            'correct': correct,
            'total': total,
        })

        return output

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        total = torch.stack([x['total'] for x in outputs]).sum()
        val_acc = correct * 100.0 / total

        tensorboard_logs = {'test_loss': test_loss,
                            'test_acc': val_acc}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    net = EncDecNetLightning(4, 2, 5, 'leakyRelu', batchSize=128)
    trainer = pl.Trainer(gpus=1, max_nb_epochs=15, progress_bar_refresh_rate=2,
                         default_save_path='../results', weights_save_path='../results/EncDecNet.pth')
    trainer.fit(net)
    trainer.test(net)
