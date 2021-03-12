import os
from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from .myDatasets import RightLaneDataset, ParallelDataset
from .myTransforms import MyTransform


class BaseDataModule(LightningDataModule):
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


class SimulatorDataModule(BaseDataModule):
    def setup(self, stage=None):
        self.dataSets['train'] = RightLaneDataset(os.path.join(self.dataPath, 'train'), self.train_transforms,
                                                  haveLabels=True)
        self.dataSets['valid'] = RightLaneDataset(os.path.join(self.dataPath, 'valid'), self.val_transforms,
                                                  haveLabels=True)
        self.dataSets['test'] = RightLaneDataset(os.path.join(self.dataPath, 'test'), self.test_transforms,
                                                 haveLabels=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['train'], batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['valid'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['test'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class TwoDomainDM(BaseDataModule):
    def setup(self, stage=None):
        self.dataSets['source'] = RightLaneDataset(os.path.join(self.dataPath, 'source'), self.train_transforms,
                                                   haveLabels=True)
        self.dataSets['targetTrain'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'train'),
                                                        self.train_transforms, haveLabels=True)
        self.dataSets['targetUnlabelled'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'unlabelled'),
                                                             self.train_transforms, haveLabels=False)
        self.dataSets['targetTest'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'test'),
                                                       self.test_transforms, haveLabels=True)

    def train_dataloader(self) -> DataLoader:
        sourceSet = self.dataSets['source']
        targetSet = self.dataSets['targetTrain']
        STSet = ConcatDataset([sourceSet, targetSet])

        source_weights = [1.0 / len(sourceSet) for _ in range(len(sourceSet))]
        target_weights = [1.0 / len(targetSet) for _ in range(len(targetSet))]
        weights = [*source_weights, *target_weights]

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(STSet), replacement=True)
        return DataLoader(STSet, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['targetTest'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class TwoDomainMMEDM(BaseDataModule):
    def setup(self, stage=None):
        self.dataSets['source'] = RightLaneDataset(os.path.join(self.dataPath, 'source'), self.train_transforms,
                                                   haveLabels=True)
        self.dataSets['targetTrain'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'train'),
                                                        self.train_transforms, haveLabels=True)
        self.dataSets['targetUnlabelled'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'unlabelled'),
                                                             self.train_transforms, haveLabels=False)
        self.dataSets['targetTest'] = RightLaneDataset(os.path.join(self.dataPath, 'target', 'test'),
                                                       self.test_transforms, haveLabels=True)

    def train_dataloader(self) -> DataLoader:
        sourceSet = self.dataSets['source']
        targetSet = self.dataSets['targetTrain']
        STSet = ConcatDataset([sourceSet, targetSet])
        parallelDataset = ParallelDataset(STSet, self.dataSets['targetUnlabelled'])
        assert len(STSet) <= len(self.dataSets['targetUnlabelled'])

        source_weights = [1.0 / len(sourceSet) for _ in range(len(sourceSet))]
        target_weights = [1.0 / len(targetSet) for _ in range(len(targetSet))]
        weights = [*source_weights, *target_weights]

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(STSet), replacement=True)
        return DataLoader(parallelDataset, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataSets['targetTest'], batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
