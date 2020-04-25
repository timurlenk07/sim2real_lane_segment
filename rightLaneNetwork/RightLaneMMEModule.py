from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset

from dataManagement.getData import getRightLaneDatasetsMME
from dataManagement.myDatasets import ParallelDataset
from models.FCDenseNet.tiramisu import FCDenseNet57Base, FCDenseNet57Classifier


def adentropy(output, lamda=1.0):
    return lamda * torch.mean(torch.sum(output * (torch.log(output + 1e-5)), 1))


class RightLaneMMEModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # All hyperparameters
        self.hparams = hparams

        # Dataset parameters
        self.sourceSet, self.targetTrainSet, self.targetUnlabelledSet, self.targetTestSet = (None for _ in range(4))
        self.STSet = None
        self.batchSize = hparams.batchSize

        # Dataset transformation parameters
        self.grayscale = hparams.grayscale
        self.newRes = (hparams.width, hparams.height)

        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.lr = hparams.learningRate
        self.decay = hparams.decay

        # Network parts
        self.featureExtractor = FCDenseNet57Base()
        self.classifier = FCDenseNet57Classifier(n_classes=2)

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x

    def transform(self, img, label):
        img = TF.to_pil_image(img)
        if self.grayscale:
            img = TF.to_grayscale(img)
        img = TF.resize(img, self.newRes, interpolation=Image.LANCZOS)
        img = TF.to_tensor(img)

        if label is not None and len(label.shape) >= 2:
            label = TF.to_pil_image(label)
            label = TF.resize(label, self.newRes, interpolation=Image.LANCZOS)
            label = TF.to_tensor(label).squeeze()
            label[label > 0.5] = 1
            label[label != 1] = 0
            label = label.long()

        return img, label

    def prepare_data(self):
        datasets = getRightLaneDatasetsMME('./data', transform=self.transform)
        self.sourceSet, self.targetTrainSet, self.targetUnlabelledSet, self.targetTestSet = datasets
        self.STSet = ConcatDataset([self.sourceSet, self.targetTrainSet])

    def train_dataloader(self):
        parallelDataset = ParallelDataset(self.STSet, self.targetUnlabelledSet)
        trainLoader = DataLoader(parallelDataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        return trainLoader

    def val_dataloader(self):
        return DataLoader(self.targetTestSet, batch_size=self.batchSize, shuffle=True, num_workers=4)

    def configure_optimizers(self):
        optimizer_g = Adam(self.featureExtractor.parameters(), lr=self.lr, weight_decay=self.decay)
        optimizer_f = Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.decay)
        return [optimizer_g, optimizer_f]

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_labelled, x_unlabelled, labels, _ = batch

        if optimizer_idx == 0:  # We are labelled optimizer -> minimize entropy
            outputs = self.forward(x_labelled)
            loss = self.criterion(outputs, labels)
        if optimizer_idx == 1:  # We are unlabelled optimizer -> maximize entropy
            outputs = self.featureExtractor.forward(x_unlabelled)
            outputs = self.classifier.forward(outputs, reverseGrad=True)
            loss = adentropy(outputs, lamda=0.1)

        output = OrderedDict({
            'loss': loss
        })
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Hálón átpropagáljuk a bemenetet, költséget számítunk
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # acc
        _, labels_hat = torch.max(outputs, 1)
        correct = labels_hat.eq(y).sum()
        total = torch.tensor(y.numel())

        output = OrderedDict({
            'val_loss': loss,
            'correct': correct,
            'total': total,
        })

        return output

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        total = torch.stack([x['total'] for x in outputs]).sum()
        val_acc = correct * 100.0 / total

        tensorboard_logs = {'val_loss': val_loss,
                            'val_acc': val_acc}
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}


def main(args):
    model = RightLaneMMEModule(hparams=args)

    # makes all trainer options available from the command line
    # trainer = pl.Trainer.from_argparse_args(args)

    # makes use of pre-defined options
    trainer = pl.Trainer(gpus=1, max_epochs=args.numEpochs, progress_bar_refresh_rate=2,
                         default_save_path='results', fast_dev_run=args.test)

    trainer.fit(model)
    trainer.save_checkpoint('./results/FCDenseNet57MME.ckpt')


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    # adds all the trainer options as default arguments (like max_epochs)
    # parser = pl.Trainer.add_argparse_args(parser)

    # parametrize the network
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)

    # parametrize training
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--learningRate', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--numEpochs', type=int, default=2)
    args = parser.parse_args()

    print(args)
    main(args)
