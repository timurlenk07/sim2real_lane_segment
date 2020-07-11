import glob
import logging
import os

import cv2
import torch
from torch.utils.data import Dataset


class RightLaneDataset(Dataset):
    def __init__(self, dataPath, transform=None, haveLabels=True):
        super().__init__()

        self.haveLabels = haveLabels

        self._input_dir = os.path.join(dataPath, 'input')
        if haveLabels:
            self._target_dir = os.path.join(dataPath, 'label')

        # In case no input or target exist, raise ValueError
        if not os.path.exists(self._input_dir) \
                or (haveLabels and not os.path.exists(self._target_dir)):
            raise ValueError(f"Directory structure under {dataPath} is not complete!")

        self._data_cnt = len(glob.glob(os.path.join(self._input_dir, '*.png')))
        if haveLabels and self._data_cnt != len(glob.glob(os.path.join(self._target_dir, '*.png'))):
            raise FileNotFoundError(f"Different input and target count encountered at {dataPath}!")

        if self._data_cnt == 0:
            logging.warning(f"No data found at {dataPath}!")

        self.transform = transform

    def __len__(self):
        return self._data_cnt

    def __getitem__(self, index):
        filename = f'{index:06d}.png'
        x = cv2.imread(os.path.join(self._input_dir, filename), cv2.IMREAD_COLOR)
        if self.haveLabels:
            y = cv2.imread(os.path.join(self._target_dir, filename), cv2.IMREAD_GRAYSCALE)
        else:
            y = torch.empty(0, dtype=torch.long)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y

    def __setitem__(self, index, value):
        filename = f'{index:06d}.png'
        if self.haveLabels:
            img, label = value
        else:
            img = value

        cv2.imwrite(os.path.join(self._input_dir, filename), img)
        if self.haveLabels:
            cv2.imwrite(os.path.join(self._target_dir, filename), label)


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
