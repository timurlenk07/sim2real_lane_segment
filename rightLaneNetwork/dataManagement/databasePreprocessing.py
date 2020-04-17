import glob
import logging
import os

import cv2


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
