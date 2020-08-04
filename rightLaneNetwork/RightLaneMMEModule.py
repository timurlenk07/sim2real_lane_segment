import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from dataManagement.myDatasets import ParallelDataset, RightLaneDataset
from dataManagement.myTransforms import MyTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier, grad_reverse


def adentropy(output, lamda=1.0):
    return lamda * torch.mean(torch.sum(output * (torch.log(output + 1e-5)), 1))


class RightLaneMMEModule(pl.LightningModule):
    def __init__(self, dataPath=None, width=160, height=120, gray=False,
                 augment=False, batch_size=32, lr=1e-3, decay=1e-4, **kwargs):
        super().__init__()

        self.dataPath = dataPath
        self.dataSets = dict()

        # Network parameters
        self.width, self.height = width, height
        self.grayscale = gray

        # Create network parts
        self.featureExtractor = FCDenseNet57Base()
        self.classifier = FCDenseNet57Classifier(n_classes=2)

        # Training parameters
        self.augment = augment
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay

        # Save hyperparameters
        self.save_hyperparameters('width', 'height', 'gray', 'augment', 'batch_size', 'lr', 'decay')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Network parameters
        parser.add_argument('--gray', action='store_true', help="Convert input image to grayscale")
        parser.add_argument('--width', type=int, default=160, help="Resize width of input images of the network")
        parser.add_argument('--height', type=int, default=120, help="Resize height of input images of the network")

        # Training parameters
        parser.add_argument('--augment', action='store_true', help="Use data augmentation on training set")
        parser.add_argument('-lr', '--learningRate', type=float, default=1e-3, help="Starting learning rate")
        parser.add_argument('--decay', type=float, default=1e-4, help="L2 weight decay value")
        parser.add_argument('-b', '--batch_size', type=int, default=32, help="Input batch size")

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def prepare_data(self):
        trainTransform = MyTransform(width=self.width, height=self.height, gray=self.grayscale, augment=self.augment)
        testTransform = MyTransform(width=self.width, height=self.height, gray=self.grayscale, augment=False)

        self.dataSets['source'] = RightLaneDataset(os.path.join(self.dataPath, 'source'), trainTransform,
                                                   haveLabels=True)
        self.dataSets['targetTrain'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'train'),
                                                        trainTransform, haveLabels=True)
        self.dataSets['targetUnlabelled'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'unlabelled'),
                                                             trainTransform, haveLabels=False)
        self.dataSets['targetTest'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'test'), testTransform,
                                                       haveLabels=True)

    def train_dataloader(self):
        sourceSet = self.dataSets['source']
        targetSet = self.dataSets['targetTrain']
        STSet = ConcatDataset([sourceSet, targetSet])
        parallelDataset = ParallelDataset(STSet, self.dataSets['targetUnlabelled'])
        assert len(STSet) <= len(self.dataSets['targetUnlabelled'])

        source_weights = [1.0 / len(sourceSet) for _ in range(len(sourceSet))]
        target_weights = [1.0 / len(targetSet) for _ in range(len(targetSet))]
        weights = [*source_weights, *target_weights]

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(STSet), replacement=True)
        return DataLoader(parallelDataset, sampler=sampler, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.dataSets['targetTest'], batch_size=self.batch_size, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizerF = SGD(self.parameters(), lr=self.lr, weight_decay=self.decay, momentum=0.9, nesterov=True)
        optimizerG = SGD([
            {'params': self.featureExtractor.parameters(), 'lr': self.lr / 10},
            {'params': self.classifier.parameters(), 'lr': self.lr / 3.1}
        ], lr=self.lr, weight_decay=self.decay, momentum=0.9, nesterov=True)
        lr_schedulerF = CosineAnnealingLR(optimizerF, T_max=25, eta_min=self.lr * 1e-3)
        lr_schedulerG = CosineAnnealingLR(optimizerG, T_max=25, eta_min=self.lr * 1e-4)
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

        # Forward propagation, loss calculation
        outputs = self.forward(x)
        loss = cross_entropy(outputs, y)

        _, labels_hat = torch.max(outputs, 1)

        weight = x.shape[0]
        output = OrderedDict({
            'loss': loss * weight,
            'acc': accuracy(labels_hat, y) * weight,
            'dice': dice_score(outputs, y) * weight,
            'iou': iou(labels_hat, y) * weight,
            'weight': weight,
        })

        return output

    def validation_epoch_end(self, outputs):
        total_weight = sum(x['weight'] for x in outputs)
        val_loss = sum([x['loss'] for x in outputs]) / total_weight
        val_acc = sum([x['acc'] for x in outputs]) / total_weight * 100.0
        val_dice = sum([x['dice'] for x in outputs]) / total_weight
        val_iou = sum([x['iou'] for x in outputs]) / total_weight * 100.0

        logs = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'step': self.current_epoch,
        }
        return {'progress_bar': logs, 'log': logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        total_weight = sum(x['weight'] for x in outputs)
        test_loss = sum([x['loss'] for x in outputs]) / total_weight
        test_acc = sum([x['acc'] for x in outputs]) / total_weight * 100.0
        test_dice = sum([x['dice'] for x in outputs]) / total_weight
        test_iou = sum([x['iou'] for x in outputs]) / total_weight * 100.0

        logs = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_dice': test_dice,
            'test_iou': test_iou,
        }
        return {'progress_bar': logs, 'log': logs}


def main(args, model_name: str, reproducible: bool, comet: bool):
    if reproducible:
        seed_everything(42)
        args.deterministic = True
        args.benchmark = True

    if args.default_root_dir is None:
        args.default_root_dir = 'results'

    if comet:
        comet_logger = pl.loggers.CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
            project_name=os.environ.get('COMET_PROJECT_NAME'),  # Optional
            experiment_name=model_name  # Optional
        )
        args.logger = comet_logger

    # Save best model
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.default_root_dir, model_name + '.ckpt'),
        save_top_k=1,
        verbose=False,
        monitor='val_iou',
        mode='max',
        prefix=model_name + str(os.getpid())
    )
    args.checkpoint_callback = model_checkpoint

    model = RightLaneMMEModule(**vars(args))
    model.load_state_dict(torch.load(args.pretrained_path))

    # Parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model)

    # Reload best model
    model = RightLaneMMEModule.load_from_checkpoint(model_checkpoint.kth_best_model_path, dataPath=args.dataPath)

    # Save checkpoint and weights
    ckpt_path = os.path.join(args.default_root_dir, model_name + '.ckpt')
    weights_path = os.path.join(args.default_root_dir, model_name + '_weights.pth')
    trainer.save_checkpoint(ckpt_path)
    torch.save(model.state_dict(), weights_path)
    if args.comet:
        comet_logger.experiment.log_model(model_name + '_ckpt', ckpt_path)
        comet_logger.experiment.log_model(model_name + '_weights', weights_path)

    # Perform testing
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Data location
    parser.add_argument('--dataPath', type=str, help="Path of database root")

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--model_name', type=str, default='mme', help='Model identifier for logging and checkpoints.')
    parser.add_argument('--reproducible', action='store_true',
                        help="Set seed to 42 and deterministic and benchmark to True.")

    # Need pretrained weights
    parser.add_argument('--pretrained_path', type=str,
                        help="This script uses pretrained weights of FCDenseNet57. Define path to weights here.")

    # Add model arguments to parser
    parser = RightLaneMMEModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args, model_name=args.model_name, reproducible=args.reproducible, comet=args.comet)
