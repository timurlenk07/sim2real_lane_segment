import functools
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations import Compose, ToGray, Resize, NoOp, HueSaturationValue, Normalize, MotionBlur, RandomSizedCrop, \
    GaussNoise, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataManagement.myDatasets import RightLaneDataset
from dataManagement.myTransforms import testTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class RightLaneModule(pl.LightningModule):
    def __init__(self, *, dataPath=None, width=160, height=120, gray=False,
                 augment=False, batchSize=32, lr=1e-3, decay=1e-4, lrRatio=1e3, **kwargs):
        super().__init__()

        self.dataPath = dataPath
        self.trainSet, self.validSet, self.testSet = None, None, None

        self.width, self.height = width, height
        self.grayscale = gray
        self.criterion = nn.CrossEntropyLoss()
        self.augment = augment
        self.batchSize = batchSize
        self.lr = lr
        self.decay = decay
        self.lrRatio = lrRatio

        # save hyperparameters
        self.save_hyperparameters('width', 'height', 'gray', 'augment', 'batchSize', 'lr', 'decay', 'lrRatio')
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
        parser.add_argument('--augment', action='store_true', help='Convert input image to grayscale')
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
        augmentations = Compose([
            HueSaturationValue(always_apply=True),
            RandomSizedCrop(min_max_height=(self.height // 2, self.height * 4), height=self.height, width=self.width,
                            w2h_ratio=self.width / self.height, always_apply=True),
            OneOf([MotionBlur(p=0.5), GaussNoise(p=0.5)], p=1),
        ])
        aug = Compose([
            augmentations if self.augment else Resize(height=self.height, width=self.width, always_apply=True),
            ToGray(always_apply=True) if self.grayscale else NoOp(always_apply=True),
            Normalize(always_apply=True),
            ToTensorV2(),
        ])

        if label is not None:
            # Binarize label
            label[label != 0] = 1

            augmented = aug(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask'].squeeze().long()
        else:
            augmented = aug(image=img)
            img = augmented['image']

        return img, label

    def prepare_data(self):
        testTransform2 = functools.partial(testTransform, width=self.width, height=self.height, gray=self.grayscale)

        self.trainSet = RightLaneDataset(os.path.join(self.dataPath, 'train'), self.transform, haveLabels=True)
        self.validSet = RightLaneDataset(os.path.join(self.dataPath, 'valid'), testTransform2, haveLabels=True)
        self.testSet = RightLaneDataset(os.path.join(self.dataPath, 'test'), testTransform2, haveLabels=True)

    def train_dataloader(self):
        return DataLoader(self.trainSet, batch_size=self.batchSize, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.testSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 25, eta_min=self.lr / self.lrRatio)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # acc
        _, labels_hat = torch.max(outputs, 1)
        train_acc = accuracy(labels_hat, y) * 100

        progress_bar = {
            'tr_acc': train_acc
        }
        logs = {
            'train_loss': loss,
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

        _, labels_hat = torch.max(outputs, 1)

        output = OrderedDict({
            'val_loss': loss,
            'acc': accuracy(labels_hat, y),
            'dice': dice_score(outputs, y),
            'iou': iou(labels_hat, y, remove_bg=True),
            'weight': y.shape[0],
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        weight_count = sum(x['weight'] for x in outputs)
        weighted_acc = torch.stack([x['acc'] * 100.0 * x['weight'] for x in outputs]).sum()
        weighted_dice = torch.stack([x['dice'] * x['weight'] for x in outputs]).sum()
        weighted_iou = torch.stack([x['iou'] * 100.0 * x['weight'] for x in outputs]).sum()
        val_acc = weighted_acc / weight_count
        val_dice = weighted_dice / weight_count
        val_iou = weighted_iou / weight_count

        tensorboard_logs = {'val_loss': val_loss,
                            'val_acc': val_acc,
                            'val_dice': val_dice,
                            'val_iou': val_iou}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        _, labels_hat = torch.max(outputs, 1)

        output = OrderedDict({
            'test_loss': loss,
            'acc': accuracy(labels_hat, y),
            'dice': dice_score(outputs, y),
            'iou': iou(labels_hat, y, remove_bg=True),
            'weight': y.shape[0],
        })

        return output

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        weight_count = sum(x['weight'] for x in outputs)
        weighted_acc = torch.stack([x['acc'] * 100.0 * x['weight'] for x in outputs]).sum()
        weighted_dice = torch.stack([x['dice'] * x['weight'] for x in outputs]).sum()
        weighted_iou = torch.stack([x['iou'] * 100.0 * x['weight'] for x in outputs]).sum()
        test_acc = weighted_acc / weight_count
        test_dice = weighted_dice / weight_count
        test_iou = weighted_iou / weight_count

        tensorboard_logs = {'test_loss': test_loss,
                            'test_acc': test_acc,
                            'test_dice': test_dice,
                            'test_iou': test_iou}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}


def main(args):
    model = RightLaneModule(**vars(args))

    if args.comet:
        comet_logger = pl.loggers.CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                                              workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
                                              project_name=os.environ.get('COMET_PROJECT_NAME'),  # Optional
                                              experiment_name='baseline'  # Optional
                                              )
        args.logger = comet_logger

    # Parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model)

    # Save checkpoint and weights
    root_dir = args.default_root_dir if args.default_root_dir is not None else 'results'
    ckpt_path = os.path.join(root_dir, 'baseline.ckpt')
    weights_path = os.path.join(root_dir, 'baseline_weights.pth')
    trainer.save_checkpoint(ckpt_path)
    torch.save(model.state_dict(), weights_path)

    # Perform testing
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')

    # Add model arguments to parser
    parser = RightLaneModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
