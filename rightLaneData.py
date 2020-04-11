import concurrent.futures
import glob
import itertools
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def video2images(directory, transform=None, deleteProcessed=False):
    logging.info(f"Managing directory: {directory}")

    # Get the list of available recordings
    annot_vids = sorted(glob.glob(os.path.join(directory, '*_annot_pp.avi')))
    orig_vids = sorted(glob.glob(os.path.join(directory, '*_orig_pp.avi')))

    # Check whether original and annotated recordings number match or not
    if len(annot_vids) != len(orig_vids):
        raise RuntimeError(f"Different number of input and target videos!")

    logging.info(f"{directory} Number of files found: {len(annot_vids)}. Taking apart video files...")

    os.makedirs(os.path.join(directory, 'orig'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'annot'), exist_ok=True)

    img_counter = 0
    # Iterate and postprocess every recording
    for vid_a, vid_o in zip(annot_vids, orig_vids):

        # Open recordings...
        cap_orig = cv2.VideoCapture(vid_o)
        cap_annot = cv2.VideoCapture(vid_a)
        if not cap_orig.isOpened() or not cap_annot.isOpened():
            unopened = cap_orig if not cap_orig.isOpened() else cap_annot
            logging.warning(f"Could not open file {unopened}! Continuing...", )
            continue

        # Check whether recordings hold the same number of frames
        if cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) != cap_annot.get(cv2.CAP_PROP_FRAME_COUNT):
            logging.warning(f"Different video length encountered at: {cap_orig}! Continuing...")
            continue

        # Produce output videos
        logging.debug(f"Processing recording: {vid_o}...")
        while cap_orig.isOpened() and cap_annot.isOpened():  # Iterate through every frame
            ret_o, frame_o = cap_orig.read()
            ret_a, frame_a = cap_annot.read()
            if not ret_o or not ret_a:
                break

            # Convert annotated to grayscale
            frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)

            if transform is not None:
                frame_o, frame_a = transform(frame_o, frame_a)

            filename = f'{img_counter:06d}.png'
            filepath_o = os.path.join(directory, 'orig', filename)
            filepath_a = os.path.join(directory, 'annot', filename)

            # Save both frames in new file
            cv2.imwrite(filepath_o, frame_o)
            cv2.imwrite(filepath_a, frame_a)

            img_counter += 1

        logging.debug(f"Processing of recording done for: {vid_o}")

        # Release VideoCapture resources
        cap_orig.release()
        cap_annot.release()

        # Delete processed videos upon request
        if deleteProcessed:
            os.remove(vid_a)
            os.remove(vid_o)

    logging.info(f"Video files taken apart in {directory}! Images generated: {img_counter}")
    return img_counter


class RightLaneImagesDataset(Dataset):
    def __init__(self, dataPath, transform=None, shouldPreprocess=False, preprocessTransform=None):
        super().__init__()

        self._input_dir = os.path.join(dataPath, 'orig')
        self._target_dir = os.path.join(dataPath, 'annot')

        # In case no input or target exist, try to do preprocessing
        if not os.path.exists(self._input_dir) or not os.path.exists(self._target_dir):
            shouldPreprocess = True

        if shouldPreprocess:
            video2images(dataPath, preprocessTransform, deleteProcessed=False)

        self._data_cnt = len(glob.glob(os.path.join(self._input_dir, '*.png')))
        if self._data_cnt != len(glob.glob(os.path.join(self._target_dir, '*.png'))):
            raise FileNotFoundError(f"Different input and target count encountered at {dataPath}!")
        if self._data_cnt == 0:
            raise FileNotFoundError(f"No data found at {dataPath}!")

        self.transform = transform

    def __len__(self):
        return self._data_cnt

    def __getitem__(self, index):
        filename = f'{index:06d}.png'
        x = cv2.imread(os.path.join(self._input_dir, filename), cv2.IMREAD_COLOR)
        y = cv2.imread(os.path.join(self._target_dir, filename), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y


def getRightLaneDatasets(dataPath, transform=None, shouldPreprocess=False, preprocessTransform=None):
    # assert dataPath is a valid path
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    train_dir = os.path.join(dataPath, "train")
    valid_dir = os.path.join(dataPath, "validation")
    test_dir = os.path.join(dataPath, "test")
    dataPaths = [train_dir, valid_dir, test_dir]

    for i, directory in enumerate(dataPaths):
        # assert directory already exist (else there would be no data)
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist!")

    transforms = list(itertools.repeat(transform, len(dataPaths)))
    shouldPreprocess = list(itertools.repeat(shouldPreprocess, len(dataPaths)))
    preprocessTransform = list(itertools.repeat(preprocessTransform, len(dataPaths)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        datasets = executor.map(RightLaneImagesDataset, dataPaths, transforms, shouldPreprocess, preprocessTransform)

    datasets = tuple(datasets)

    return datasets


# Adatbetöltőket készítő függvény
def getDataLoaders(datasets, batchSize=128):
    trainLoader = torch.utils.data.DataLoader(datasets[0], batch_size=batchSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(datasets[1], batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(datasets[2], batch_size=batchSize, shuffle=True)
    return trainLoader, validLoader, testLoader


class SavedTransform:
    def __init__(self, grayscale: bool, newRes: tuple = None):
        self.grayscale = grayscale
        self.newRes = newRes

    def __call__(self, img, label):
        if img is not None:
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.newRes is not None:
                img = cv2.resize(img, self.newRes)

        if label is not None:
            if self.newRes is not None:
                label = cv2.resize(label, self.newRes)

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LoadedTransform:
    def __init__(self, grayscale: bool, newRes: tuple = None):
        self.grayscale = grayscale
        self.newRes = newRes

    def __call__(self, img, label):
        if img is not None:
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.newRes is not None:
                img = cv2.resize(img, self.newRes, interpolation=cv2.INTER_LANCZOS4)

            if self.grayscale:
                img = np.expand_dims(img, -1)

            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img.float() / 255

        if label is not None:
            if self.newRes is not None:
                label = cv2.resize(label, self.newRes, interpolation=cv2.INTER_LANCZOS4)

                # y should be binary
                _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

            label = torch.from_numpy(label).long() / 255

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    # Adatbázis építés
    datasetParams = {
        'transform': SavedTransform(grayscale=False, newRes=(160, 120)),
        'shouldPreprocess': False,
        'preprocessTransform': LoadedTransform(grayscale=True, newRes=(80, 60)),
    }
    datasets = getRightLaneDatasets('./data',
                                    transform=LoadedTransform(grayscale=False, newRes=(160, 120)),
                                    shouldPreprocess=True,
                                    preprocessTransform=SavedTransform(grayscale=True, newRes=(320, 240)),
                                    )
    trainSet, validSet, testSet = datasets
    print(f"Dataset lengths: {len(trainSet)}, {len(validSet)}, {len(testSet)}")

    x, y = trainSet[0]
    print(f"Dataset returned data with input: {x.shape}, and output: {y.shape}")

    # Lekérjük, hogy rendelkezésre áll-e GPU
    haveCuda = torch.cuda.is_available()

    if haveCuda:
        x, y = x.cuda(), y.cuda()

    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))
