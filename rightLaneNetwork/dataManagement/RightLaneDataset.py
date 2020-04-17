import glob
import os

import cv2
from torch.utils.data import Dataset


class RightLaneDataset(Dataset):
    def __init__(self, dataPath, transform=None, labelled=True):
        super().__init__()

        self._labelled = labelled

        self._input_dir = os.path.join(dataPath, 'input')
        if labelled:
            self._target_dir = os.path.join(dataPath, 'label')

        # In case no input or target exist, raise ValueError
        if not os.path.exists(self._input_dir) or (labelled and not os.path.exists(self._target_dir)):
            raise ValueError(f"Directory structure under {dataPath} is not complete!")

        self._data_cnt = len(glob.glob(os.path.join(self._input_dir, '*.png')))
        if labelled and self._data_cnt != len(glob.glob(os.path.join(self._target_dir, '*.png'))):
            raise FileNotFoundError(f"Different input and target count encountered at {dataPath}!")

        # TODO: remove suppression or checking
        if False and self._data_cnt == 0:
            raise FileNotFoundError(f"No data found at {dataPath}!")

        self.transform = transform

    def __len__(self):
        return self._data_cnt

    def __getitem__(self, index):
        filename = f'{index:06d}.png'
        x = cv2.imread(os.path.join(self._input_dir, filename), cv2.IMREAD_COLOR)
        if self._labelled:
            y = cv2.imread(os.path.join(self._target_dir, filename), cv2.IMREAD_GRAYSCALE)
        else:
            y = None

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y
