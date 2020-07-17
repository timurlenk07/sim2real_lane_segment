import functools
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from albumentations import (
    Compose, ToGray, Resize, NoOp, HueSaturationValue, Normalize, MotionBlur, RandomSizedCrop, GaussNoise, OneOf
)
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from dataManagement.myDatasets import RightLaneDataset
from dataManagement.myTransforms import testTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class RightLaneSTModule(pl.LightningModule):
    def __init__(self, *, dataPath=None, width=160, height=120, gray=False,
                 augment=False, batchSize=32, lr=1e-3, decay=1e-4, lrRatio=1e3, **kwargs):
        super().__init__()

        self.dataPath = dataPath
        self.sourceSet, self.targetTrainSet, self.targetUnlabelledSet, self.targetTestSet = None, None, None, None

        self.width, self.height = width, height
        self.grayscale = gray
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
        parser.add_argument('--dataPath', type=str, help="Data root path")

        # parametrize the network
        parser.add_argument('--gray', action='store_true', help='Convert input image to grayscale')
        parser.add_argument('-wd', '--width', type=int, default=160, help='Resized input image width')
        parser.add_argument('-hg', '--height', type=int, default=120, help='Resized input image height')

        # Training hyperparams
        parser.add_argument('--augment', action='store_true', help='Convert input image to grayscale')
        parser.add_argument('-lr', '--learningRate', type=float, default=1e-3, help='Base learning rate')
        parser.add_argument('--decay', type=float, default=1e-4, help='L2 weight decay value')
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

        self.sourceSet = RightLaneDataset(os.path.join(self.dataPath, 'source'), self.transform, haveLabels=True)
        self.targetTrainSet = RightLaneDataset(os.path.join(self.dataPath, 'target', 'train'),
                                               self.transform, haveLabels=True)
        self.targetUnlabelledSet = RightLaneDataset(os.path.join(self.dataPath, 'target', 'unlabelled'),
                                                    self.transform, haveLabels=False)
        self.targetTestSet = RightLaneDataset(os.path.join(self.dataPath, 'target', 'test'), testTransform2,
                                              haveLabels=True)

    def train_dataloader(self):
        STSet = ConcatDataset([self.sourceSet, self.targetTrainSet])
        source_weights = [1.0 / len(self.sourceSet) for _ in range(len(self.sourceSet))]
        target_weights = [1.0 / len(self.targetTrainSet) for _ in range(len(self.targetTrainSet))]
        weights = [*source_weights, *target_weights]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(STSet), replacement=True)
        return DataLoader(STSet, sampler=sampler, batch_size=self.batchSize, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.targetTestSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 25, eta_min=self.lr / self.lrRatio)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = cross_entropy(outputs, y)

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
        loss = cross_entropy(outputs, y)

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
        weighted_acc = torch.stack([x['acc'] * x['weight'] for x in outputs]).sum()
        weighted_dice = torch.stack([x['dice'] * x['weight'] for x in outputs]).sum()
        weighted_iou = torch.stack([x['iou'] * x['weight'] for x in outputs]).sum()
        val_acc = weighted_acc / weight_count * 100.0
        val_dice = weighted_dice / weight_count
        val_iou = weighted_iou / weight_count * 100.0

        logs = {'val_loss': val_loss,
                'val_acc': val_acc,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'step': self.current_epoch}
        return {'progress_bar': logs, 'log': logs}


def main(args):
    if args.reproducible:
        import numpy as np
        np.random.seed(42)
        torch.manual_seed(42)
        args.deterministic = True
        args.benchmark = True

    model = RightLaneSTModule(**vars(args))

    if args.default_root_dir is None:
        args.default_root_dir = 'results'

    if args.comet:
        comet_logger = pl.loggers.CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                                              workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
                                              project_name=os.environ.get('COMET_PROJECT_NAME'),  # Optional
                                              experiment_name='S_and_T'  # Optional
                                              )
        args.logger = comet_logger

    # Save best model
    args.checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.default_root_dir, 'sandt.ckpt'),
        save_top_k=1,
        verbose=False,
        monitor='val_iou',
        mode='max',
        prefix=str(os.getpid())
    )

    # Parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model)

    # Reload best model
    model.load_from_checkpoint(args.checkpoint_callback.kth_best_model)

    # Save checkpoint and weights
    ckpt_path = os.path.join(args.default_root_dir, 'sandt.ckpt')
    weights_path = os.path.join(args.default_root_dir, 'sandt_weights.pth')
    trainer.save_checkpoint(ckpt_path)
    torch.save(model.state_dict(), weights_path)
    if args.comet:
        comet_logger.experiment.log_model('sandt_ckpt', ckpt_path)
        comet_logger.experiment.log_model('sandt_weights', weights_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--reproducible', action='store_true', help="Set seed to 42 and deterministic to True.")

    # Add model arguments to parser
    parser = RightLaneSTModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
