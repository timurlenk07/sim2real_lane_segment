import shutil
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.EncDecNet import EncDecNet
from rightLaneData import getRightLaneDatasets, LoadedTransform, SavedTransform


class RightLaneModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.dataSets = None

        self.grayscale = hparams.grayscale
        self.criterion = nn.CrossEntropyLoss()
        self.batchSize = hparams.batchSize
        self.lr = hparams.learningRate
        self.decay = hparams.decay
        self.lrRatio = hparams.lrRatio

        inFeat = 3 if not self.grayscale else 1
        self.net = EncDecNet(nFeat=hparams.nFeat, nLevels=hparams.nLevels, kernelSize=hparams.kernelSize,
                             nLinType=hparams.nLinType, bNorm=hparams.batchNorm, dropOut=hparams.dropOut, inFeat=inFeat)

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        self.dataSets = getRightLaneDatasets('./data',
                                             transform=LoadedTransform(grayscale=self.grayscale, newRes=(160, 120)),
                                             shouldPreprocess=False,
                                             preprocessTransform=SavedTransform(grayscale=self.grayscale,
                                                                                newRes=(160, 120)),
                                             )

    def train_dataloader(self):
        return DataLoader(self.dataSets[0], batch_size=self.batchSize, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataSets[1], batch_size=self.batchSize, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataSets[2], batch_size=self.batchSize, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
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


def main(args):
    if args.doPreprocess:
        shutil.rmtree('./data/train/orig')
        shutil.rmtree('./data/train/annot')
        shutil.rmtree('./data/validation/orig')
        shutil.rmtree('./data/validation/annot')
        shutil.rmtree('./data/test/orig')
        shutil.rmtree('./data/test/annot')

    model = RightLaneModule(hparams=args)

    # makes all trainer options available from the command line
    # trainer = pl.Trainer.from_argparse_args(args)

    # makes use of pre-defined options
    trainer = pl.Trainer(gpus=1, max_epochs=args.numEpochs, progress_bar_refresh_rate=2,
                         default_save_path='./results', weights_save_path='./results/EncDecNet.pth')

    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    # adds all the trainer options as default arguments (like max_epochs)
    # parser = pl.Trainer.add_argparse_args(parser)

    # parametrize the network
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--doPreprocess', action='store_true')

    parser.add_argument('--nFeat', type=int, default=16)
    parser.add_argument('--nLevels', type=int, default=2)
    parser.add_argument('--kernelSize', type=int, default=3)
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--nLinType', type=str, default='relu')
    parser.add_argument('--batchNorm', type=bool, default=True)
    parser.add_argument('--dropOut', type=float, default=0.3)
    parser.add_argument('--learningRate', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--lrRatio', type=float, default=1000)
    parser.add_argument('--numEpochs', type=int, default=2)
    args = parser.parse_args()

    print(args)
    main(args)
