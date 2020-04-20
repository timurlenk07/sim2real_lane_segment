import argparse
import concurrent.futures
import glob
import logging
import os
from itertools import zip_longest

import cv2


def video2images(directory, transform=None, haveLabels=True, deleteProcessed=False):
    logging.info(f"Managing directory: {directory}")

    # Get the list of available recordings
    input_vids = sorted(glob.glob(os.path.join(directory, '*_orig_pp.avi')))
    label_vids = sorted(glob.glob(os.path.join(directory, '*_annot_pp.avi')))

    # Check whether original and annotated recordings number match or not
    if haveLabels and len(input_vids) != len(label_vids):
        raise RuntimeError(f"Different number of input and target videos!")

    os.makedirs(os.path.join(directory, 'input'), exist_ok=True)
    if haveLabels:
        os.makedirs(os.path.join(directory, 'label'), exist_ok=True)

    if len(input_vids) == 0:
        logging.info(f"{directory}: No data found. You might want to check if this is an intended behaviour.")
        return 0

    if not haveLabels:
        logging.info(f"{directory}: No labels found. You might want to check if this is an intended behaviour.")

    logging.info(f"{directory} Number of files found: {len(input_vids)}. Taking apart video files...")

    img_counter = 0
    # Iterate and postprocess every recording
    for input_vid, label_vid in zip_longest(input_vids, label_vids):
        assert label_vid is not None if haveLabels else True

        # Open recordings...
        input_cap = cv2.VideoCapture(input_vid)
        if haveLabels:
            label_cap = cv2.VideoCapture(label_vid)

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
            if haveLabels:
                ret_l, label_frame = label_cap.read()
            if not ret_i or (haveLabels and not ret_l):
                break

            if haveLabels:
                # Convert annotated to grayscale
                label_frame = cv2.cvtColor(label_frame, cv2.COLOR_BGR2GRAY)

            if transform is not None:
                if haveLabels:
                    input_frame, label_frame = transform(input_frame, label_frame)
                else:
                    input_frame, _ = transform(input_frame, None)

            # Save both frames in new file
            filename = f'{img_counter:06d}.png'
            filepath_o = os.path.join(directory, 'input', filename)
            cv2.imwrite(filepath_o, input_frame)
            if haveLabels:
                filepath_a = os.path.join(directory, 'label', filename)
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


def createRightLaneDatabase(dataPath, shouldPreprocess=False, preprocessTransform=None):
    # Check data is available, ie. file structure is as is required

    # Main directory
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    # Set directories
    train_dir = os.path.join(dataPath, "train")
    valid_dir = os.path.join(dataPath, "validation")
    test_dir = os.path.join(dataPath, "test")
    dataPaths = [train_dir, valid_dir, test_dir]
    exists, missing = checkDirsExist(dataPaths)
    if not exists:
        raise FileNotFoundError(f"Directories: {missing} do not exist, hence no data is available!")

    # Check if post preprocess folders are available, if not, try to preprocess
    if not shouldPreprocess:
        target_dirs = [os.path.join(dataPath, 'input') for dataPath in dataPaths]
        target_dirs.extend([os.path.join(dataPath, 'label') for dataPath in dataPaths])
        exists, missing = checkDirsExist(target_dirs)
        if not exists:
            shouldPreprocess = True
            dataPaths = [os.path.split(directory)[0] for directory in missing]

    if shouldPreprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(video2images, dataPath, preprocessTransform, True, False)
                       for dataPath in dataPaths]
            concurrent.futures.as_completed(futures)


def createRightLaneDatabaseMME(dataPath, shouldPreprocess=False, preprocessTransform=None):
    # Check data is available, ie. file structure is as is required

    # Main directory
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Directory {dataPath} does not exist!")

    # Domain directories
    source_path = os.path.join(dataPath, "source")
    target_path = os.path.join(dataPath, 'target')
    dataPaths = [source_path, target_path]
    exists, missing = checkDirsExist(dataPaths)
    if not exists:
        raise FileNotFoundError(f"Directories: {missing} do not exist, hence no data is available!")

    # Sub-domain directories
    target_train_path = os.path.join(target_path, 'train')
    target_unlabelled_path = os.path.join(target_path, 'unlabelled')
    target_test_path = os.path.join(target_path, 'test')
    dataPaths = [source_path, target_train_path, target_unlabelled_path, target_test_path]
    labelled = [True, True, False, True]  # Will be used later, easier to track it here
    exists, missing = checkDirsExist(dataPaths)
    if not exists:
        raise FileNotFoundError(f"Directories: {missing} do not exist, hence no data is available!")

    # Check if post preprocess folders are available, if not, try to preprocess
    if not shouldPreprocess:
        exists, missing = checkDirsExist([os.path.join(dataPath, 'input') for dataPath in dataPaths])
        if not exists:
            shouldPreprocess = True
            dataPaths = [os.path.split(directory)[0] for directory in missing]

    if shouldPreprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(video2images, dataPath, preprocessTransform, haveLabels, True)
                       for dataPath, haveLabels in zip(dataPaths, labelled)]
            concurrent.futures.as_completed(futures)


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
                label = cv2.resize(label, self.newRes)

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimax', action='store_true')
    parser.add_argument('--dataPath', type=str, default="./realData")
    parser.add_argument('--shouldPreprocess', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    args = parser.parse_args()

    newRes = (args.width, args.height) if args.resize else None
    transform = GrayscaleResizeTransform(grayscale=args.grayscale, newRes=newRes)

    if args.minimax:
        createRightLaneDatabaseMME(args.dataPath, args.shouldPreprocess, transform)
    else:
        createRightLaneDatabase(args.dataPath, args.shouldPreprocess, transform)
