import glob
import logging
import os
from argparse import ArgumentParser
from random import shuffle
import numpy as np

import cv2

parser = ArgumentParser()
parser.add_argument('-dp', '--delete_processed', action='store_true')
parser.add_argument('-cd', '--clear_data', action='store_true')
parser.add_argument('-id', '--input_dir', default=os.path.join(os.getcwd(), 'recordings'))
parser.add_argument('-od', '--output_dir', default=os.path.join(os.getcwd(), 'data'))
args = parser.parse_args()

logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)

if args.clear_data:
    try:
        from shutil import rmtree

        rmtree(args.output_dir)
    except FileNotFoundError:
        pass


# Binarization algorithm, with a given original and annotated pair of image
def process_classes(img_orig: np.ndarray, img_ant: np.ndarray):
    img_orig = img_orig.astype(np.int16)
    img_ant = img_ant.astype(np.int16)
    img_diff = img_ant - img_orig

    leftLane = img_diff[:, :, 0] > 0
    rightLane = img_diff[:, :, 1] > 0
    obstacles = img_diff[:, :, 2] > 0
    categories = [leftLane, rightLane, obstacles]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    categories = [x.astype(np.uint8) for x in categories]
    categories = [cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel) for x in categories]
    categories = [cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel) for x in categories]
    categories = tuple((x.astype(np.bool8) for x in categories))

    leftLane, rightLane, obstacles = categories

    res_gray = np.zeros(img_orig.shape[:-1], np.uint8)
    res_gray[rightLane] = 64 # Right lane
    res_gray[leftLane] = 128  # Left lane
    res_gray[obstacles] = 192 # Obstacles

    return res_gray


# Get the list of available recordings
annot_raw_list = sorted(glob.glob(os.path.join(args.input_dir, '*_annot.avi')))
orig_raw_list = sorted(glob.glob(os.path.join(args.input_dir, '*_orig.avi')))

# Check whether original and annotated recordings number match or not
assert len(annot_raw_list) == len(orig_raw_list), "Length mismatch! No postprocess performed."

raw_list = list(zip(orig_raw_list, annot_raw_list))
shuffle(raw_list)

# Create output dir structure
input_dir = os.path.join(args.output_dir, 'input')
os.makedirs(input_dir, exist_ok=True)
label_dir = os.path.join(args.output_dir, 'label')
os.makedirs(label_dir, exist_ok=True)

# Check starting video number
vid_counter = 0

# Iterate and postprocess every recording
for orig_fp, annot_fp in raw_list:
    filename = f'{vid_counter:06d}.avi'
    inputFile = os.path.join(input_dir, filename)
    labelFile = os.path.join(label_dir, filename)
    while os.path.exists(inputFile) or os.path.exists(labelFile):
        vid_counter += 1
        filename = f'{vid_counter:06d}.avi'
        inputFile = os.path.join(input_dir, filename)
        labelFile = os.path.join(label_dir, filename)
    logging.debug(f"InputFile: {inputFile}, labelFile: {labelFile}")

    # Open recordings...
    cap_orig = cv2.VideoCapture(orig_fp)
    cap_annot = cv2.VideoCapture(annot_fp)
    if not cap_orig.isOpened() or not cap_annot.isOpened():
        logging.warning("Could not open files! Continuing...")
        continue

    # Check whether recordings hold the same number of frames
    if cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) != cap_annot.get(cv2.CAP_PROP_FRAME_COUNT):
        logging.warning("Different video length encountered! Continuing...")
        logging.debug("orig frames: %i, annot frames: %i" % (
            cap_orig.get(cv2.CAP_PROP_FRAME_COUNT), cap_annot.get(cv2.CAP_PROP_FRAME_COUNT)))
        continue

    # Open VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    fps = 30
    framesize = (640, 480)
    isColor = True
    vWriter_input = cv2.VideoWriter(inputFile, fourcc, fps, framesize, isColor)

    isColor = False
    vWriter_label = cv2.VideoWriter(labelFile, fourcc, fps, framesize, isColor)
    
    if not vWriter_input.isOpened() or not vWriter_label.isOpened():
        logging.warning("Could not open video writers! Continuing...")
        vWriter_input.release()
        vWriter_label.release()
        continue

    # Produce output videos
    logging.info(f"Processing recording nr. {vid_counter}...")
    while cap_orig.isOpened() and cap_annot.isOpened():  # Iterate through every frame
        ret_o, frame_o = cap_orig.read()
        ret_a, frame_a = cap_annot.read()
        if not ret_o or not ret_a:
            break

        # Postprocess original recording: needs nothing
        vWriter_input.write(frame_o)

        # Postprocess annotated frame: binarize it
        annot_binary = process_classes(frame_o, frame_a)
        vWriter_label.write(annot_binary)

    logging.info(f"Processing of recording nr. {vid_counter} done.")

    # Release writer resources
    vWriter_input.release()
    vWriter_label.release()

if args.delete_processed:
    try:
        from shutil import rmtree

        rmtree(args.input_dir)
    except FileNotFoundError:
        pass

logging.info("Post-processing finished!")
