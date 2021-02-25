import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataManagement.dataModules import SimulatorDataModule
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


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
        # self.log('step', self.current_epoch)

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

    data = SimulatorDataModule(dataPath=args.dataPath, augment=args.augment, batch_size=args.batch_size, num_workers=8)
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
    parser = SimulatorDataModule.add_argparse_args(parser)
    parser = RightLaneModule.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    main(args, model_name=args.model_name, reproducible=args.reproducible, comet=args.comet, wandb=args.wandb)
