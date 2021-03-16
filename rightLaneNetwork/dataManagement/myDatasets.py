import glob
import logging
import os

import cv2
import torch
from torch.utils.data import Dataset


class RightLaneDataset(Dataset):
    def __init__(self, dataPath, transform=None, haveLabels=True, *,
                 loadIntoMemory=False, onLoadTransform=None):
        super().__init__()

        self.haveLabels = haveLabels

        self._input_dir = os.path.join(dataPath, 'input')
        if haveLabels:
            self._target_dir = os.path.join(dataPath, 'label')

        # In case no input or target exist, raise ValueError
        if not os.path.exists(self._input_dir) \
                or (haveLabels and not os.path.exists(self._target_dir)):
            raise ValueError(f"Directory structure under {dataPath} is not complete!")

        self._dataPaths = {'input': glob.glob(os.path.join(self._input_dir, '*.png'))}
        self._data_cnt = len(self._dataPaths['input'])
        if self._data_cnt == 0:
            logging.warning(f"No data found at {dataPath}!")

        if haveLabels:
            self._dataPaths['target'] = glob.glob(os.path.join(self._target_dir, '*.png'))
            if self._data_cnt != len(self._dataPaths['target']):
                raise FileNotFoundError(f"Different input and target count encountered at {dataPath}!")

        self.transform = transform

        self._data = None
        if loadIntoMemory:
            self._preloadData(onLoadTransform)

    def __len__(self):
        return self._data_cnt

    def __getitem__(self, index):
        if self._data is not None:
            x = self._data['input'][index]
            if self.haveLabels:
                y = self._data['target'][index]
        else:
            x = cv2.imread(self._dataPaths['input'][index], cv2.IMREAD_COLOR)
            if self.haveLabels:
                y = cv2.imread(self._dataPaths['target'][index], cv2.IMREAD_GRAYSCALE)

        if not self.haveLabels:
            y = torch.empty(0, dtype=torch.long)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y

    def __setitem__(self, index, value):
        if self.haveLabels:
            img, label = value
        else:
            img = value

        cv2.imwrite(self._dataPaths['input'][index], img)
        if self.haveLabels:
            cv2.imwrite(self._dataPaths['target'][index], label)

    def _preloadData(self, onLoadTransform):
        self._data = {'input': [cv2.imread(p, cv2.IMREAD_COLOR) for p in self._dataPaths['input']]}
        if self.haveLabels:
            self._data['target'] = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in self._dataPaths['target']]


class ParallelDataset(Dataset):
    def __init__(self, dsA, dsB):
        super().__init__()

        self.dsA, self.dsB = dsA, dsB

    def __len__(self):
        return len(self.dsA)

    def __getitem__(self, index):
        x1, y1 = self.dsA[index % len(self.dsA)]
        x2, y2 = self.dsB[index % len(self.dsB)]

        return x1, x2, y1, y2


class UnbalancedDataset(Dataset):
    def __init__(self, longer, shorter):
        super().__init__()

        self.longer, self.shorter = longer, shorter

    def __len__(self):
        return len(self.longer)

    def __getitem__(self, index):
        x1, y1 = self.longer[index % len(self.longer)]
        x2, y2 = self.shorter[index % len(self.shorter)]

        return (x1, x2), (y1, y2)
