import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataManagement.myDatasets import RightLaneDataset
from dataManagement.myTransforms import MyTransform
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


class RightLaneDataModule(LightningDataModule):
    def __init__(self, *, dataPath=None, width=160, height=120, gray=False, augment=False, batch_size=1, num_workers=1):
        super().__init__(
            train_transforms=MyTransform(width=width, height=height, gray=gray, augment=augment),
            val_transforms=MyTransform(width=width, height=height, gray=gray, augment=False),
            test_transforms=MyTransform(width=width, height=height, gray=gray, augment=False),
            dims=(width, height, 1 if gray else 3)
        )
        self.dataSets = dict()
        self.dataPath = dataPath if dataPath is not None else os.getcwd()
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        dmGroup = parser.add_argument_group('DataModule', 'Parameters defining data handling')

        dmGroup.add_argument('--gray', action='store_true', help="Convert input image to grayscale")
        dmGroup.add_argument('--width', type=int, default=160, help="Resize width of input images of the network")
        dmGroup.add_argument('--height', type=int, default=120, help="Resize height of input images of the network")
        dmGroup.add_argument('--augment', action='store_true', help="Use data augmentation on training set")
        dmGroup.add_argument('-b', '--batch_size', type=int, default=32, help="Input batch size")

        return parser

    def setup(self, stage=None):
        self.dataSets['train'] = RightLaneDataset(os.path.join(self.dataPath, 'train'), self.train_transforms,
                                                  haveLabels=True)
        self.dataSets['valid'] = RightLaneDataset(os.path.join(self.dataPath, 'valid'), self.val_transforms,
                                                  haveLabels=True)
        self.dataSets['test'] = RightLaneDataset(os.path.join(self.dataPath, 'test'), self.test_transforms,
                                                 haveLabels=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['train'], batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['valid'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['test'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)


class RightLaneModule(LightningModule):
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
        argGroup = parser.add_argument_group('LightningModule', 'Parameters defining network training')

        argGroup.add_argument('-lr', '--learningRate', type=float, default=1e-3, help="Starting learning rate")
        argGroup.add_argument('--decay', type=float, default=1e-4, help="L2 weight decay value")
        argGroup.add_argument('--lrRatio', type=float, default=1000,
                              help="Ratio of maximum and minimum of learning rate for cosine LR scheduler")

        return parser

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward propagation, loss calculation
        outputs = self.forward(x)
        loss = cross_entropy(outputs, y)

        # Accuracy calculation
        _, labels_hat = torch.max(outputs, 1)
        train_acc = accuracy(labels_hat, y) * 100

        self.log('tr_loss', loss)
        self.log('tr_acc', train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate_batch(batch)

    def validation_epoch_end(self, outputs):
        logs = self.summarize_evaluation_results(outputs)
        self.log('val_loss', logs['loss'])
        self.log('val_acc', logs['acc'], prog_bar=True, logger=True)
        self.log('val_dice', logs['dice'])
        self.log('val_iou', logs['iou'], prog_bar=True, logger=True)
        #self.log('step', self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.evaluate_batch(batch)

    def test_epoch_end(self, outputs):
        logs = self.summarize_evaluation_results(outputs)
        self.log('test_loss', logs['loss'])
        self.log('test_acc', logs['acc'])
        self.log('test_dice', logs['dice'])
        self.log('test_iou', logs['iou'])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = CosineAnnealingLR(optimizer, 25, eta_min=self.lr / self.lrRatio)
        return [optimizer], [scheduler]

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

    def summarize_evaluation_results(self, outputs):
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


def main(args, model_name: str, reproducible: bool, comet: bool, wandb: bool):
    if reproducible:
        seed_everything(42)
        args.deterministic = True
        args.benchmark = True

    if comet:
        from pytorch_lightning.loggers import CometLogger
        comet_logger = CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
            project_name=os.environ.get('COMET_PROJECT_NAME'),  # Optional
            experiment_name=model_name  # Optional
        )
        args.logger = comet_logger
    if wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project=os.environ.get('WANDB_PROJECT_NAME'), log_model=True, sync_step=True)
        args.logger = wandb_logger

    if args.default_root_dir is None:
        args.default_root_dir = 'results'

    # Save best model
    model_checkpoint = ModelCheckpoint(
        filename=model_name + '_{epoch}',
        save_top_k=1,
        monitor='val_iou',
        mode='max',
    )
    args.checkpoint_callback = model_checkpoint

    data = RightLaneDataModule(dataPath=args.dataPath, augment=True, batch_size=64, num_workers=8)
    model = RightLaneModule(lr=args.learningRate, lrRatio=args.lrRatio, decay=args.decay, num_cls=4)

    # Parse all trainer options available from the command line
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data)

    # Reload best model
    model = RightLaneModule.load_from_checkpoint(model_checkpoint.best_model_path, dataPath=args.dataPath, num_cls=4)

    # Upload weights
    if comet:
        comet_logger.experiment.log_model(model_name + '_weights', model_checkpoint.best_model_path)

    # Perform testing
    trainer.test(model, datamodule=data)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Data location
    parser.add_argument('--dataPath', type=str, help="Path of database root")

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--wandb', action='store_true', help='Define flag in order to use WandB as logger.')
    parser.add_argument('--model_name', type=str, default='baseline',
                        help='Model identifier for logging and checkpoints.')
    parser.add_argument('--reproducible', action='store_true',
                        help="Set seed to 42 and deterministic and benchmark to True.")

    # Add model arguments to parser
    parser = RightLaneDataModule.add_argparse_args(parser)
    parser = RightLaneModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    main(args, model_name=args.model_name, reproducible=args.reproducible, comet=args.comet, wandb=args.wandb)
