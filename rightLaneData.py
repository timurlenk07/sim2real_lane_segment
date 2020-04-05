import concurrent.futures
import glob
import logging
import os

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def video2images(directory, newRes):
    logging.info(f"Managing directory: {directory}")

    # Get the list of available recordings
    annot_vids = sorted(glob.glob(os.path.join(directory, '*_annot_pp.avi')))
    orig_vids = sorted(glob.glob(os.path.join(directory, '*_orig_pp.avi')))

    # Check whether original and annotated recordings number match or not
    assert len(annot_vids) == len(orig_vids)

    logging.info(f"{directory} Number of files found: {len(annot_vids)}. Taking apart video files...")

    os.makedirs(os.path.join(directory, 'orig'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'annot'), exist_ok=True)

    img_counter = 0
    vid_counter = 0
    # Iterate and postprocess every recording
    for vid_a, vid_o in zip(annot_vids, orig_vids):

        # Open recordings...
        cap_orig = cv2.VideoCapture(vid_o)
        cap_annot = cv2.VideoCapture(vid_a)
        if not cap_orig.isOpened() or not cap_annot.isOpened():
            logging.warning(f"{directory} Could not open file nr. {vid_counter}! Continuing...", )
            continue

        # Check whether recordings hold the same number of frames
        if cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) != cap_annot.get(cv2.CAP_PROP_FRAME_COUNT):
            logging.warning(f"{directory} Different video length encountered in video nr. {vid_counter}! Continuing...")
            continue

        # Produce output videos
        logging.debug(f"{directory} Processing recording nr. {vid_counter}...")
        while cap_orig.isOpened() and cap_annot.isOpened():  # Iterate through every frame
            ret_o, frame_o = cap_orig.read()
            ret_a, frame_a = cap_annot.read()
            if not ret_o or not ret_a:
                break

            frame_o = cv2.resize(frame_o, newRes, interpolation=cv2.INTER_LANCZOS4)
            frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            frame_a = cv2.resize(frame_a, newRes, interpolation=cv2.INTER_LANCZOS4)

            # y should be binary
            _, frame_a = cv2.threshold(frame_a, 127, 255, cv2.THRESH_BINARY)

            filename = str(img_counter).zfill(6) + '.png'
            filepath_o = os.path.join(directory, 'orig', filename)
            filepath_a = os.path.join(directory, 'annot', filename)
            # Save both frames in new file
            cv2.imwrite(filepath_o, frame_o)
            cv2.imwrite(filepath_a, frame_a)

            img_counter += 1

        logging.debug(f"{directory} Processing of recording nr. {vid_counter} done.")
        vid_counter += 1

        # Release VideoCapture resources
        cap_orig.release()
        cap_annot.release()

        os.remove(vid_a)
        os.remove(vid_o)

    logging.info(f"{directory} Video files taken apart! Images generated: {img_counter}")


class RightLaneImagesDataset(Dataset):
    def __init__(self, dataPath, shouldPreprocess=False, resolution=(160, 120)):

        self.input_dir = os.path.join(dataPath, 'orig')
        self.target_dir = os.path.join(dataPath, 'annot')

        if not os.path.exists(self.input_dir) or not os.path.exists(self.target_dir):
            shouldPreprocess = True

        if shouldPreprocess:
            video2images(dataPath, resolution)

        self.data_cnt = len(glob.glob(os.path.join(self.input_dir, '*.png')))
        if self.data_cnt != len(glob.glob(os.path.join(self.target_dir, '*.png'))):
            raise FileNotFoundError(f"Different input and target count encountered!")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, index):
        filename = f'{index:06d}.png'
        x = cv2.imread(os.path.join(self.input_dir, filename))
        y = cv2.imread(os.path.join(self.target_dir, filename), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            x = self.transform(x)

        y = torch.from_numpy(y).long() / 255
        return x, y

    def setTransform(self, transform):
        self.transform = transform


def getRightLaneDatasets(dataPath):
    # assert dataPath is a valid path
    assert os.path.exists(dataPath)

    train_dir = os.path.join(dataPath, "train")
    valid_dir = os.path.join(dataPath, "validation")
    test_dir = os.path.join(dataPath, "test")
    dirs = [train_dir, valid_dir, test_dir]

    # assert dir in dirs exist
    for directory in dirs:
        assert os.path.exists(directory)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        datasets = executor.map(RightLaneImagesDataset, dirs)

    return tuple(datasets)


# Adatbetöltőket készítő függvény
def getDataLoaders(datasets, batchSize=128):
    trainLoader = torch.utils.data.DataLoader(datasets[0], batch_size=batchSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(datasets[1], batch_size=batchSize, shuffle=True)
    testLoader = torch.utils.data.DataLoader(datasets[2], batch_size=batchSize, shuffle=True)
    return trainLoader, validLoader, testLoader


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    # Adatbázis építés
    trainSet, validSet, testSet = getRightLaneDatasets('./data')

    print(len(trainSet), len(validSet), len(testSet))

    x, y = trainSet[0]
    print(x.shape, y.shape)

    # Lekérjük, hogy rendelkezésre áll-e GPU
    haveCuda = torch.cuda.is_available()

    if haveCuda:
        x, y = x.cuda(), y.cuda()
        print(x.shape, y.shape)

    x, y = trainSet[0]
    print(torch.mean(x))
    print(torch.min(x), torch.min(y))
    print(torch.max(x), torch.max(y))
