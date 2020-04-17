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
