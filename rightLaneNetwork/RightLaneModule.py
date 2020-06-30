from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations import Compose, ToGray, Resize, Blur, NoOp
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataManagement.getData import getRightLaneDatasets
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class RightLaneModule(pl.LightningModule):
    def __init__(self, *, dataPath=None, width=160, height=120, gray=False,
                 batchSize=32, lr=1e-3, decay=1e-4, lrRatio=1e3, **kwargs):
        super().__init__()

        self.dataPath = dataPath
        self.trainSet, self.validSet, self.testSet = (None for _ in range(3))

        self.width, self.height = width, height
        self.grayscale = gray
        self.criterion = nn.CrossEntropyLoss()
        self.batchSize = batchSize
        self.lr = lr
        self.decay = decay
        self.lrRatio = lrRatio

        # save hyperparameters
        self.save_hyperparameters('width', 'height', 'gray', 'batchSize', 'lr', 'decay', 'lrRatio')
        print(f"The model has the following hyperparameters:")
        print(self.hparams)

        # Network parts
        self.featureExtractor = FCDenseNet57Base()
        self.classifier = FCDenseNet57Classifier(n_classes=2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Data location
        parser.add_argument('--dataPath', type=str, default='./data')

        # parametrize the network
        parser.add_argument('--gray', action='store_true', help='Convert input image to grayscale')
        parser.add_argument('-wd', '--width', type=int, default=160, help='Resized input image width')
        parser.add_argument('-hg', '--height', type=int, default=120, help='Resized input image height')

        # Training hyperparams
        parser.add_argument('-lr', '--learningRate', type=float, default=1e-3, help='Base learning rate')
        parser.add_argument('--decay', type=float, default=1e-4,
                            help='L2 weight decay value')
        parser.add_argument('--lrRatio', type=float, default=1000)
        parser.add_argument('-b', '--batchSize', type=int, default=32, help='Input batch size')

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def transform(self, img, label=None):
        aug = Compose([
            Blur(p=0.5),
            Resize(height=self.height, width=self.width, always_apply=True),
            ToGray(always_apply=True) if self.grayscale else NoOp(always_apply=True),
            ToTensor(),
        ])

        if label is not None:
            # Binarize label
            label[label > 0] = 255

            augmented = aug(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask'].squeeze().long()
        else:
            augmented = aug(image=img)
            img = augmented['image']

        return img, label

    def prepare_data(self):
        dataSets = getRightLaneDatasets(self.dataPath, transform=self.transform)
        self.trainSet, self.validSet, self.testSet = dataSets

    def train_dataloader(self):
        return DataLoader(self.trainSet, batch_size=self.batchSize, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 20, eta_min=self.lr / self.lrRatio)
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
        acc = Accuracy()(labels_hat, y)

        output = OrderedDict({
            'test_loss': loss,
            'acc': acc,
            'weight': y.shape[0],
        })

        return output

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        weight_count = sum(x['weight'] for x in outputs)
        weighted_acc = torch.stack([x['acc'] * 100.0 * x['weight'] for x in outputs]).sum()
        val_acc = weighted_acc / weight_count

        tensorboard_logs = {'test_loss': test_loss,
                            'test_acc': val_acc}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}


def main(args):
    model = RightLaneModule(**vars(args))

    # parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model)
    trainer.save_checkpoint(args.ckpt_save_path)
    if args.weights_save_path is not None:
        torch.save(model.state_dict(), args.weights_save_path)
    trainer.test(model)


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1  # Do not allow the use of more than one GPUs

    parser = ArgumentParser()

    parser.add_argument('--ckpt_save_path', type=str, default='./results/FCDenseNet57.ckpt')
    # parser.add_argument('--weights_save_path', type=str, default='./results/FCDenseNet57weights.pth')
    parser = RightLaneModule.add_model_specific_args(parser)

    # adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
