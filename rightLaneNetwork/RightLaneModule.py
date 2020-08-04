import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataManagement.myDatasets import RightLaneDataset
from dataManagement.myTransforms import MyTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class RightLaneModule(pl.LightningModule):
    def __init__(self, *, dataPath=None, width=160, height=120, gray=False,
                 augment=False, batch_size=32, lr=1e-3, decay=1e-4, lrRatio=1e3, **kwargs):
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
        self.lrRatio = lrRatio

        # Save hyperparameters
        self.save_hyperparameters('width', 'height', 'gray', 'augment', 'batch_size', 'lr', 'decay', 'lrRatio')

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
        parser.add_argument('--lrRatio', type=float, default=1000,
                            help="Ratio of maximum and minimum of learning rate for cosine LR scheduler")
        parser.add_argument('-b', '--batch_size', type=int, default=32, help="Input batch size")

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def prepare_data(self):
        trainTransform = MyTransform(width=self.width, height=self.height, gray=self.grayscale, augment=self.augment)
        testTransform = MyTransform(width=self.width, height=self.height, gray=self.grayscale, augment=False)

        self.dataSets['train'] = RightLaneDataset(os.path.join(self.dataPath, 'train'), trainTransform, haveLabels=True)
        self.dataSets['valid'] = RightLaneDataset(os.path.join(self.dataPath, 'valid'), testTransform, haveLabels=True)
        self.dataSets['test'] = RightLaneDataset(os.path.join(self.dataPath, 'test'), testTransform, haveLabels=True)

    def train_dataloader(self):
        return DataLoader(self.dataSets['train'], batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.dataSets['valid'], batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.dataSets['test'], batch_size=self.batch_size, shuffle=False, num_workers=8)

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

    if comet:
        comet_logger = pl.loggers.CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
            project_name=os.environ.get('COMET_PROJECT_NAME'),  # Optional
            experiment_name=model_name  # Optional
        )
        args.logger = comet_logger

    if args.default_root_dir is None:
        args.default_root_dir = 'results'

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

    model = RightLaneModule(**vars(args))

    # Parse all trainer options available from the command line
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model)

    # Reload best model
    model = RightLaneModule.load_from_checkpoint(model_checkpoint.kth_best_model_path, dataPath=args.dataPath)

    # Save checkpoint and weights
    ckpt_path = os.path.join(args.default_root_dir, model_name + '.ckpt')
    weights_path = os.path.join(args.default_root_dir, model_name + '_weights.pth')
    trainer.save_checkpoint(ckpt_path)
    torch.save(model.state_dict(), weights_path)
    if comet:
        comet_logger.experiment.log_model(model_name + '_ckpt', ckpt_path)
        comet_logger.experiment.log_model(model_name + '_weights', weights_path)

    # Perform testing
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Data location
    parser.add_argument('--dataPath', type=str, help="Path of database root")

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--model_name', type=str, default='baseline',
                        help='Model identifier for logging and checkpoints.')
    parser.add_argument('--reproducible', action='store_true',
                        help="Set seed to 42 and deterministic and benchmark to True.")

    # Add model arguments to parser
    parser = RightLaneModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args, model_name=args.model_name, reproducible=args.reproducible, comet=args.comet)
