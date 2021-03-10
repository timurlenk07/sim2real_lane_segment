import os
import sys
from argparse import ArgumentParser

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataManagement.dataModules import SimulatorDataModule, TwoDomainDM, BaseDataModule, TwoDomainMMEDM
from trainingModules.MMETrainingModule import MMETrainingModule
from trainingModules.SimpleTrain import SimpleTrainModule
from trainingModules.TrainingBase import TrainingBase


def main(args, train_type: str, model_name: str, reproducible: bool, comet: bool, wandb: bool):
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
        save_top_k=1, save_weights_only=True,
        monitor='val_iou', mode='max',
    )
    args.checkpoint_callback = model_checkpoint

    if train_type == 'sim':
        data = SimulatorDataModule(dataPath=args.dataPath, augment=args.augment, batch_size=args.batch_size,
                                   num_workers=8)
        model = SimpleTrainModule(lr=args.learningRate, lrRatio=args.lrRatio, decay=args.decay, num_cls=4)
    elif train_type == 'st':
        data = TwoDomainDM(dataPath=args.dataPath, augment=args.augment, batch_size=args.batch_size, num_workers=8)
        model = SimpleTrainModule(lr=args.learningRate, lrRatio=args.lrRatio, decay=args.decay, num_cls=4)
    elif train_type == 'mme':
        data = TwoDomainMMEDM(dataPath=args.dataPath, augment=args.augment, batch_size=args.batch_size, num_workers=8)
        model = MMETrainingModule(lr=args.learningRate, lrRatio=args.lrRatio, decay=args.decay, num_cls=4)
        model.load_state_dict(torch.load(args.pretrained_path))
    else:
        raise RuntimeError(f"Not recognizable training type: {train_type}")

    # Parse all trainer options available from the command line
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data)

    # Perform testing
    trainer.test()

    # Upload weights
    if comet:
        comet_logger.experiment.log_model(model_name + '_weights', model_checkpoint.best_model_path)

    # Save best weights (save-weights_only does not work)
    dirname = os.path.dirname(model_checkpoint.best_model_path)
    torch.save(model.state_dict(), os.path.join(dirname, 'best_weights.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--trainType', choices=['sim', 'st', 'mme'], help="Type of training method")
    parser.add_argument('--dataPath', type=str, help="Path of database root", required=True)

    # MME training needs pretrained weights
    parser.add_argument('--pretrained_path', type=str, required='--trainType=mme' in sys.argv,
                        help="MME training uses pretrained weights. Use this to define path to it.")

    parser.add_argument('--model_name', type=str, default='baseline',
                        help='Model identifier for logging and checkpoints.')
    parser.add_argument('--reproducible', action='store_true',
                        help="Set seed to 42 and deterministic and benchmark to True.")

    parser.add_argument('--comet', action='store_true', help='Define flag in order to use Comet.ml as logger.')
    parser.add_argument('--wandb', action='store_true', help='Define flag in order to use WandB as logger.')

    # Add data and model arguments to parser
    parser = BaseDataModule.add_argparse_args(parser)
    parser = TrainingBase.add_model_specific_args(parser)

    # Adds all the trainer options as default arguments (like max_epochs)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    main(args,
         train_type=args.trainType, model_name=args.model_name, reproducible=args.reproducible,
         comet=args.comet, wandb=args.wandb)
