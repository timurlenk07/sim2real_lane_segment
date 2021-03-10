from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy

from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class TrainingBase(LightningModule):
    def __init__(self, lr=1e-3, decay=1e-4, lrRatio=1e3, num_cls=2):
        super().__init__()
        self.save_hyperparameters('lr', 'decay', 'lrRatio')

        # Create network parts
        self.featureExtractor = FCDenseNet57Base()
        self.classifier = FCDenseNet57Classifier(n_classes=num_cls)

        # Training parameters
        self.lr = lr
        self.decay = decay
        self.lrRatio = lrRatio

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        argGroup = parser.add_argument_group('TrainingModule', 'Parameters defining network training')

        argGroup.add_argument('-lr', '--learningRate', type=float, default=1e-3, help="Starting learning rate")
        argGroup.add_argument('--decay', type=float, default=1e-4, help="L2 weight decay value")
        argGroup.add_argument('--lrRatio', type=float, default=1000,
                              help="Ratio of maximum and minimum of learning rate for cosine LR scheduler")

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def validation_step(self, batch, batch_idx):
        return self.evaluate_batch(batch)

    def validation_epoch_end(self, outputs):
        logs = self.summarize_evaluation_results(outputs)
        self.log('val_loss', logs['loss'])
        self.log('val_acc', logs['acc'], prog_bar=True, logger=True)
        self.log('val_dice', logs['dice'])
        self.log('val_iou', logs['iou'], prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self.evaluate_batch(batch)

    def test_epoch_end(self, outputs):
        logs = self.summarize_evaluation_results(outputs)
        self.log('test_loss', logs['loss'])
        self.log('test_acc', logs['acc'])
        self.log('test_dice', logs['dice'])
        self.log('test_iou', logs['iou'])

    def evaluate_batch(self, batch):
        x, y = batch

        # Forward propagation, loss calculation
        outputs = self.forward(x)
        loss = cross_entropy(outputs, y)

        _, labels_hat = torch.max(outputs, 1)

        weight = x.shape[0]
        return {
            'loss': loss * weight,
            'acc': accuracy(labels_hat, y) * weight,
            'dice': dice_score(outputs, y) * weight,
            'iou': iou(labels_hat, y) * weight,
            'weight': weight,
        }

    @staticmethod
    def summarize_evaluation_results(outputs):
        total_weight = sum(x['weight'] for x in outputs)
        loss = sum([x['loss'] for x in outputs]) / total_weight
        acc = sum([x['acc'] for x in outputs]) / total_weight * 100.0
        dice = sum([x['dice'] for x in outputs]) / total_weight
        iou = sum([x['iou'] for x in outputs]) / total_weight * 100.0

        return {
            'loss': loss,
            'acc': acc,
            'dice': dice,
            'iou': iou,
        }
