import math
import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from dataManagement.dataModules import BaseDataModule, TwoDomainMMEDM
from trainingModules.MMETrainingModule import MMETrainingModule
from trainingModules.TrainingBase import TrainingBase

NUM_CLS = 3


def trainWithTune(config, checkpoint_dir=None, datamodule=None, num_epochs=10, num_gpus=0):
    trainer = Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "val_loss",
                    "mean_accuracy": "val_acc",
                    "mean_iou": "val_iou",
                },
                filename="checkpoint",
                on="validation_end")
        ])

    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(os.path.join(checkpoint_dir, "checkpoint"), map_location=lambda storage, loc: storage)
        model = MMETrainingModule._load_model_state(ckpt, lr=10**config['log_lr'], lrRatio=10**config['log_lrRatio'],
                                                    decay=10**config['log_decay'], num_cls=NUM_CLS)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = MMETrainingModule(lr=10**config['log_lr'], lrRatio=10**config['log_lrRatio'],
                                  decay=10**config['log_decay'], num_cls=NUM_CLS)

    trainer.fit(model, datamodule=datamodule)


def main(args, reproducible: bool):
    if reproducible:
        seed_everything(42)

    datamodule = TwoDomainMMEDM(dataPath=args.dataPath, augment=True, batch_size=32, num_workers=8)

    config = {
        "log_lr": tune.uniform(-4, -2),
        "log_lrRatio": tune.uniform(-3, 0),
        "log_decay": tune.uniform(-8, -1),
    }

    search_alg = BayesOptSearch(
        metric='mean_iou',
        mode='max',
    )

    scheduler = ASHAScheduler(
        grace_period=25,
    )

    reporter = CLIReporter(
        parameter_columns=["log_lr", "log_lrRatio", "log_decay"],
        metric_columns=["loss", "mean_iou", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            trainWithTune,
            datamodule=datamodule, num_epochs=175, num_gpus=1,
        ),
        resources_per_trial={
            "cpu": 5,
            "gpu": 0.5,
        },
        metric="mean_iou",
        mode="max",
        config=config,
        num_samples=20,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune_minimax_segmenter")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataPath', type=str, help="Path of database root", required=True)

    # MME training needs pretrained weights
    parser.add_argument('--pretrained_path', type=str, required='--trainType=mme' in sys.argv,
                        help="MME training uses pretrained weights. Use this to define path to it.")

    parser.add_argument('--reproducible', action='store_true',
                        help="Set seed to 42 and deterministic and benchmark to True.")

    # Add data and model arguments to parser
    parser = BaseDataModule.add_argparse_args(parser)
    parser = TrainingBase.add_model_specific_args(parser)

    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    main(args, reproducible=args.reproducible)
