import argparse
import glob
import logging
import os
import shutil
from random import shuffle, seed

import cv2
from tqdm import tqdm


def videos2images(directory, transform=None, haveLabels=True, deleteProcessed=False):
    logging.info(f"Managing directory: {directory}")

    input_dir = os.path.join(directory, 'input')
    label_dir = os.path.join(directory, 'label') if haveLabels else None

    if not os.path.isdir(input_dir) or (haveLabels and not os.path.isdir(label_dir)):
        raise FileNotFoundError(f"Unexpected directory structure!")

    # Get the list of available recordings
    input_vids = sorted(glob.glob(os.path.join(input_dir, '*.avi')))
    label_vids = sorted(glob.glob(os.path.join(label_dir, '*.avi'))) if haveLabels else None

    # Check whether original and annotated recordings number match or not
    if haveLabels and len(input_vids) != len(label_vids):
        raise RuntimeError(f"Different number of input and target videos!")

    if len(input_vids) == 0:
        logging.info(f"{directory}: No data found. You might want to check if this is an intended behaviour.")
        return 0

    if not haveLabels:
        logging.info(f"{directory}: No labels given. You might want to check if this is an intended behaviour.")

    logging.info(f"{directory} Number of files found: {len(input_vids)}. Taking apart video files...")

    img_counter = 0
    # Iterate and postprocess every recording
    for input_vid, label_vid in tqdm(zip(input_vids, label_vids)):
        assert label_vid is not None if haveLabels else True

        # Open recordings...
        input_cap = cv2.VideoCapture(input_vid)
        label_cap = cv2.VideoCapture(label_vid) if haveLabels else None

        if not input_cap.isOpened() or (haveLabels and not label_cap.isOpened()):
            unopened = input_vid if not input_cap.isOpened() else label_vid
            logging.warning(f"Could not open file {unopened}! Continuing...", )
            continue

        # Check whether recordings hold the same number of frames
        if haveLabels and input_cap.get(cv2.CAP_PROP_FRAME_COUNT) != label_cap.get(cv2.CAP_PROP_FRAME_COUNT):
            logging.warning(f"Different video length encountered at: {input_vid}! Continuing...")
            continue

        # Produce output videos
        logging.debug(f"Processing recording: {input_vid}...")
        while input_cap.isOpened():  # Iterate through every frame
            ret_i, input_frame = input_cap.read()
            (ret_l, label_frame) = label_cap.read() if haveLabels else (None, None)
            if not ret_i or (haveLabels and not ret_l):
                break

            if haveLabels:
                # Convert label to grayscale
                label_frame = cv2.cvtColor(label_frame, cv2.COLOR_BGR2GRAY)

            if transform is not None:
                if haveLabels:
                    input_frame, label_frame = transform(input_frame, label_frame)
                else:
                    input_frame, _ = transform(input_frame, None)

            # Save both frames in new file
            filename = f'{img_counter:06d}.png'
            filepath_o = os.path.join(input_dir, filename)
            cv2.imwrite(filepath_o, input_frame)
            if haveLabels:
                filepath_a = os.path.join(label_dir, filename)
                cv2.imwrite(filepath_a, label_frame)

            img_counter += 1

        logging.debug(f"Processing of recording done for: {input_vid}")

        # Release VideoCapture resources
        input_cap.release()
        label_cap.release()

        # Delete processed videos upon request
        if deleteProcessed:
            os.remove(input_vid)
            os.remove(label_vid)

    logging.info(f"{directory}: Video files taken apart! Images generated: {img_counter}")
    return img_counter


def checkDirsExist(directories):
    missingDirs = []
    for directory in directories:
        if not os.path.exists(directory):
            missingDirs.append(directory)

    return len(missingDirs) == 0, missingDirs


def createRightLaneDatabase(dataPath, preprocessTransform=None, useSingleSet=False):
    # Check data is available, ie. file structure is as is required

    # Main directory
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    # Test expected directories
    input_dir = os.path.join(dataPath, "input")
    label_dir = os.path.join(dataPath, "label")
    dataPaths = [input_dir, label_dir]
    exists, missing = checkDirsExist(dataPaths)
    if not exists:
        raise FileNotFoundError(f"Directories: {missing} do not exist, hence no data is available!")

    # Take apart videos
    videos2images(dataPath, preprocessTransform, True, True)

    if useSingleSet:
        return

    # Create train, valid and test set
    train_ratio = 0.7
    test_ratio = 0.15

    input_imgs = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    label_imgs = sorted(glob.glob(os.path.join(label_dir, '*.png')))
    assert len(input_imgs) == len(label_imgs), "Input and label image count is not the same!"
    imgs = list(zip(input_imgs, label_imgs))
    shuffle(imgs)

    train_end_idx = int(round(len(imgs) * train_ratio))
    test_start_idx = int(round(len(imgs) * (1 - test_ratio)))
    assert train_end_idx < test_start_idx, "Train end is beyond test start; probably too few data is available!"

    setPaths = [os.path.join(dataPath, set_name) for set_name in ['train', 'valid', 'test']]
    imgSets = [imgs[:train_end_idx], imgs[train_end_idx:test_start_idx], imgs[test_start_idx:]]
    assert sum([len(imgSet) for imgSet in imgSets]) == len(
        imgs), "Not the same amount of images in sets and merged one!"

    for imgSet, setPath in zip(imgSets, setPaths):
        os.makedirs(os.path.join(setPath, 'input'))
        os.makedirs(os.path.join(setPath, 'label'))
        for i, (input_img, label_img) in enumerate(imgSet):
            filename = f'{i:06d}.png'
            shutil.move(input_img, os.path.join(setPath, 'input', filename))
            shutil.move(label_img, os.path.join(setPath, 'label', filename))

    shutil.rmtree(input_dir)
    shutil.rmtree(label_dir)


def preprocessRealDB(dataPath, preprocessTransform=None, train_ratio=0.7):
    # TODO: make use of preprocessTransform
    # Check data is available, ie. file structure is as is required

    # Main directory
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    # Sub-directories
    input_dir = os.path.join(dataPath, 'input')
    label_dir = os.path.join(dataPath, 'label')
    unlabelled_dir = os.path.join(dataPath, 'unlabelled')
    exists, missing = checkDirsExist([input_dir, label_dir, unlabelled_dir])
    if not exists:
        raise FileNotFoundError(f"Directories: {missing} do not exist, hence no data is available!")

    # Create train and test set
    input_imgs = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    label_imgs = sorted(glob.glob(os.path.join(label_dir, '*.png')))
    assert len(input_imgs) == len(label_imgs), "Input and label image count is not the same!"
    imgs = list(zip(input_imgs, label_imgs))
    shuffle(imgs)

    train_end_idx = int(round(len(imgs) * train_ratio))

    setPaths = [os.path.join(dataPath, set_name) for set_name in ['train', 'test']]
    imgSets = [imgs[:train_end_idx], imgs[train_end_idx:]]
    assert sum([len(imgSet) for imgSet in imgSets]) == len(
        imgs), "Not the same amount of images in sets and merged one!"

    for imgSet, setPath in zip(imgSets, setPaths):
        os.makedirs(os.path.join(setPath, 'input'))
        os.makedirs(os.path.join(setPath, 'label'))
        for i, (input_img, label_img) in enumerate(imgSet):
            filename = f'{i:06d}.png'
            shutil.move(input_img, os.path.join(setPath, 'input', filename))
            shutil.move(label_img, os.path.join(setPath, 'label', filename))

    shutil.move(unlabelled_dir, os.path.join(dataPath, '.temp'))
    shutil.move(os.path.join(dataPath, '.temp'), os.path.join(unlabelled_dir, 'input'))

    shutil.rmtree(input_dir)
    shutil.rmtree(label_dir)


class GrayscaleResizeTransform:
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
                label = cv2.resize(label, self.newRes, interpolation=cv2.INTER_NEAREST)

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbType', choices=['sim', 'real'], required=True)
    parser.add_argument('--single_sim_dir', action='store_true')
    parser.add_argument('--dataPath', type=str, default="./realData")
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    args = parser.parse_args()
    seed(42)

    newRes = (args.width, args.height) if args.resize else None
    transform = GrayscaleResizeTransform(grayscale=args.grayscale, newRes=newRes)

    assert 0 < args.train_ratio <= 1

    if args.dbType == 'real':
        preprocessRealDB(args.dataPath, transform, args.train_ratio)
    elif args.dbType == 'sim':
        createRightLaneDatabase(args.dataPath, transform, args.single_sim_dir)
