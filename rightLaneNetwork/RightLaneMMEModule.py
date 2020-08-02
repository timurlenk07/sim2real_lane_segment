import functools
import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import ray
import torch
from albumentations import (
    Compose, ToGray, Resize, NoOp, HueSaturationValue, Normalize, MotionBlur, RandomSizedCrop, GaussNoise, OneOf
)
from albumentations.pytorch import ToTensorV2
from hyperopt import hp
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch.nn.functional import cross_entropy
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from dataManagement.myDatasets import ParallelDataset, RightLaneDataset
from dataManagement.myTransforms import testTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier, grad_reverse


def adentropy(output, lamda=1.0):
    return lamda * torch.mean(torch.sum(output * (torch.log(output + 1e-5)), 1))


class RightLaneMMEModule(pl.LightningModule):
    def __init__(self, *,
                 dataPath=None, width=160, height=120, gray=False, augment=False, batchSize=32,
                 optim1='SGD', optim2='SGD', momentum1=0.9, momentum2=0.9, sched1='StepLR', sched2='StepLR',
                 lr=1e-3, lr_ratio1=1, lr_ratio2=1e-1, lr_sched_ratio=1e-4, decay=1e-4, **kwargs):
        super().__init__()

        # Dataset parameters
        self.dataPath = dataPath
        self.sourceSet, self.targetTrainSet, self.targetUnlabelledSet, self.targetTestSet = None, None, None, None

        # Dataset transformation parameters
        self.grayscale = gray
        self.augment = augment
        self.height, self.width = height, width

        # Training parameters
        self.batchSize = batchSize
        self.optim1 = optim1
        self.optim2 = optim2
        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.sched1 = sched1
        self.sched2 = sched2
        self.lr = lr
        self.lr_ratio1 = lr_ratio1
        self.lr_ratio2 = lr_ratio2
        self.lr_sched_ratio = lr_sched_ratio
        self.decay = decay

        # Save hyperparameters
        self.save_hyperparameters('width', 'height', 'gray', 'augment', 'batchSize',
                                  'optim1', 'optim2', 'momentum1', 'momentum2', 'sched1', 'sched2',
                                  'lr', 'lr_ratio1', 'lr_ratio2',
                                  'lr_sched_ratio', 'decay')

        # Network parts
        self.featureExtractor = FCDenseNet57Base()
        self.classifier = FCDenseNet57Classifier(n_classes=2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Data location
        parser.add_argument('--dataPath', type=str)

        # parametrize the network
        parser.add_argument('--gray', action='store_true', help='Convert input image to grayscale')
        parser.add_argument('-wd', '--width', type=int, default=160, help='Resized input image width')
        parser.add_argument('-hg', '--height', type=int, default=120, help='Resized input image height')

        # Training hyperparams
        parser.add_argument('--augment', action='store_true', help='Convert input image to grayscale')
        parser.add_argument('-lr', '--learningRate', type=float, default=1e-3, help='Base learning rate')
        parser.add_argument('--decay', type=float, default=1e-4,
                            help='L2 weight decay value')
        parser.add_argument('-b', '--batchSize', type=int, default=32, help='Input batch size')

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def transform(self, img, label=None):
        augmentations = Compose([
            HueSaturationValue(p=0.5),
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

        if label is not None and len(label.shape) >= 2:
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
                                                    testTransform2, haveLabels=False)
        self.targetTestSet = RightLaneDataset(os.path.join(self.dataPath, 'target', 'test'), testTransform2,
                                              haveLabels=True)

    def train_dataloader(self):
        STSet = ConcatDataset([self.sourceSet, self.targetTrainSet])
        parallelDataset = ParallelDataset(STSet, self.targetUnlabelledSet)
        assert len(STSet) <= len(self.targetUnlabelledSet)

        source_weights = [1.0 / len(self.sourceSet) for _ in range(len(self.sourceSet))]
        target_weights = [1.0 / len(self.targetTrainSet) for _ in range(len(self.targetTrainSet))]
        weights = [*source_weights, *target_weights]

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(STSet), replacement=True)
        return DataLoader(parallelDataset, batch_size=self.batchSize, sampler=sampler, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.targetTestSet, batch_size=self.batchSize, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        if self.optim1 == 'SGD':
            optimizerF = SGD(self.parameters(), lr=self.lr, weight_decay=self.decay, momentum=self.momentum1,
                             nesterov=True)
        elif self.optim1 == 'Adam':
            optimizerF = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)

        if self.optim2 == 'SGD':
            optimizerG = SGD([
                {'params': self.featureExtractor.parameters(), 'lr': self.lr * self.lr_ratio1},
                {'params': self.classifier.parameters(), 'lr': self.lr * self.lr_ratio2}
            ], lr=self.lr, weight_decay=self.decay, momentum=self.momentum2, nesterov=True)
        elif self.optim2 == 'Adam':
            optimizerG = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)

        if self.sched1 == 'StepLR':
            lr_schedulerF = StepLR(optimizerF, step_size=1, gamma=self.lr_sched_ratio ** (1.0 / 140))  # max_epochs root
        elif self.sched1 == 'CAWR':
            lr_schedulerF = CosineAnnealingWarmRestarts(optimizerF, T_0=20, T_mult=2,
                                                        eta_min=self.lr * self.lr_sched_ratio)
        elif self.sched1 == 'CALR':
            lr_schedulerF = CosineAnnealingLR(optimizerF, T_max=20, eta_min=self.lr * self.lr_sched_ratio)

        if self.sched2 == 'StepLR':
            lr_schedulerG = StepLR(optimizerG, step_size=1, gamma=self.lr_sched_ratio ** (1.0 / 140))  # max_epochs root
        elif self.sched2 == 'CAWR':
            lr_schedulerG = CosineAnnealingWarmRestarts(optimizerG, T_0=20, T_mult=2,
                                                        eta_min=self.lr * min(self.lr_ratio1,
                                                                              self.lr_ratio2) * self.lr_sched_ratio)
        elif self.sched2 == 'CALR':
            lr_schedulerG = CosineAnnealingLR(optimizerG, T_max=20,
                                              eta_min=self.lr * min(self.lr_ratio1,
                                                                    self.lr_ratio2) * self.lr_sched_ratio)

        return [optimizerG, optimizerF], [lr_schedulerG, lr_schedulerF]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x_labelled, x_unlabelled, labels, _ = batch

        if optimizer_idx == 0:  # We are unlabelled optimizer -> maximize entropy
            outputs = self.featureExtractor(x_unlabelled)
            outputs = grad_reverse(outputs)
            outputs = self.classifier(outputs)
            loss = adentropy(outputs, lamda=0.1)
        if optimizer_idx == 1:  # We are labelled optimizer -> minimize entropy
            outputs = self.featureExtractor(x_labelled)
            outputs = self.classifier(outputs)
            loss = cross_entropy(outputs, labels)

        output = OrderedDict({
            'loss': loss
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
        weight_count = sum(x['weight'] for x in outputs)
        weighted_loss = torch.stack([x['val_loss'] * x['weight'] for x in outputs]).sum()
        weighted_acc = torch.stack([x['acc'] * x['weight'] for x in outputs]).sum()
        weighted_dice = torch.stack([x['dice'] * x['weight'] for x in outputs]).sum()
        weighted_iou = torch.stack([x['iou'] * x['weight'] for x in outputs]).sum()
        val_loss = weighted_loss / weight_count
        val_acc = weighted_acc / weight_count * 100.0
        val_dice = weighted_dice / weight_count
        val_iou = weighted_iou / weight_count * 100.0

        logs = {'val_loss': val_loss,
                'val_acc': val_acc,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'step': self.current_epoch}
        return {'progress_bar': logs, 'log': logs}


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        # print(f"val_loss is {trainer.callback_metrics['val_loss'].item()}")
        tune.report(
            val_loss=trainer.callback_metrics['val_loss'].item(),
            val_acc=trainer.callback_metrics["val_acc"].item(),
            val_dice=trainer.callback_metrics["val_dice"].item(),
            val_iou=trainer.callback_metrics["val_iou"].item(),
            epoch=trainer.callback_metrics['step'],
        )


def trainMME(config, trainer, args):
    if args.reproducible:
        seed_everything(42)
        args.deterministic = True
    module_params = vars(args)
    module_params.update(config)
    module_params['sched1'] = 'CALR'
    model = RightLaneMMEModule(**module_params)
    model.load_state_dict(torch.load(args.pretrained_path))

    trainer.fit(model)


def main(args):
    if args.reproducible:
        seed_everything(42)
        args.deterministic = True

    if args.default_root_dir is None:
        args.default_root_dir = 'results'

    if args.comet:
        comet_logger = pl.loggers.CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            experiment_name='MME_hyperopt',  # Optional
        )
        args.logger = comet_logger

    args.dataPath = os.path.join(os.getcwd(), args.dataPath)
    args.pretrained_path = os.path.join(os.getcwd(), args.pretrained_path)
    args.callbacks = [TuneReportCallback()]

    # Parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    space = {
        'augment': hp.choice('augment', [True, False]),
        'momentum1': hp.choice('momentum1', [0.8, 0.9, 0.95, 0.99]),
        'momentum2': hp.choice('momentum2', [0.8, 0.9, 0.95, 0.99]),
        'sched2': hp.choice('sched2', ['StepLR', 'CALR']),
        'lr': hp.choice('lr', [1e-3 * (10 ** (i / 2)) for i in range(7)]),
        # 1e-5 to 1; Like loguniform, step by sqrt(10)
        'lr_ratio1': hp.choice('lr_ratio1', [1e-3 * (10 ** (i / 2)) for i in range(7)]),  # 1e-3 to 1
        'lr_ratio2': hp.choice('lr_ratio2', [1e-3 * (10 ** (i / 2)) for i in range(7)]),  # 1e-3 to 1
        'lr_sched_ratio': hp.choice('lr_sched_ratio', [1e-5 * (10 ** (i / 2)) for i in range(7)]),  # 1e-5 to 1e-2
    }

    suggester = HyperOptSearch(
        space=space,
        metric='val_iou',
        mode='max',
        n_initial_points=2,
    )

    scheduler = ASHAScheduler(
        metric="val_iou",
        mode="max",
        max_t=args.max_epochs,
        grace_period=min(20, args.max_epochs),
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["val_loss", "val_acc", "val_dice", 'val_iou', 'epoch']
    )

    ray.init(webui_host='127.0.0.1')
    tune.run(
        functools.partial(
            trainMME,
            trainer=trainer,
            args=args,
        ),
        resources_per_trial={"cpu": 5, "gpu": 1},
        search_alg=suggester,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mme_asha",
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--reproducible', action='store_true', help="Set seed to 42 and deterministic to True.")

    # Need pretrained weights
    parser.add_argument('--pretrained_path', type=str,
                        help='This script uses pretrained weights of FCDenseNet57. Define path to weights here.')

    # Add model arguments to parser
    parser = RightLaneMMEModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
