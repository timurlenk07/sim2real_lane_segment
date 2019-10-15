"""
This script processes the original and annotated video files exported from the simulator. The original and annotated file naming scheme takes the form of:

(sequence_nr)_(orig|annot).avi

The results of the script are preprocessed AND separated video files according to the following logic:
 - the first [ratio] size of the original video goes to the /training directory
 - the last [1-ratio] size of the original video goes to the /test directory
 - the first [ratio] size of the annotated video goes to the /validation directory
 - the last [1-ratio] size of the annotated video gets discarded
"""

import glob
import sys
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', default=0.5, type=float, help='')
parser=parser.parse_args()

ratio = parser.ratio
print("Training/Testing dataset ratio set to: {}%".format(ratio))

# Binarization algorithm, with a given original and annotated pair of image
def binarize_a(img_orig, img_ant):
    
    img_hsv = cv2.cvtColor(img_ant,cv2.COLOR_BGR2HSV)
    
    lowerBound = (10, 0, 0); #HSV
    upperBound = (170, 255, 255); #HSV
    
    mask = cv2.inRange(img_hsv, lowerBound, upperBound)
    mask = ~mask
    
    result = mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

def binarize_b(img_orig, img_ant):
    img_diff = img_orig - img_ant
    
    res_gray = cv2.cvtColor(img_diff,cv2.COLOR_BGR2GRAY)
    res_gray[res_gray > 0] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    result = cv2.morphologyEx(res_gray, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

# Wrapper function variable; select here the one you want to use
binarize = binarize_b

# Get the list of available recordings
annot_raw_list = glob.glob('recordings\*_annot.avi')
orig_raw_list = glob.glob('recordings\*_orig.avi')

# Check whether original and annotated recordings number match or not
if len(annot_raw_list) != len(orig_raw_list):
    print("Length mismatch! No postprocess performed.")
    sys.exit()

# Iterate and postprocess every recording
for i in range(len(orig_raw_list)):
    # Open recordings...
    cap_orig = cv2.VideoCapture(orig_raw_list[i])
    cap_annot = cv2.VideoCapture(annot_raw_list[i])
    if not cap_orig.isOpened() or not cap_annot.isOpened():
        print("Could not open files! Continuing...")
        continue
    
    # Check whether recordings hold the same number of frames
    if cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) != cap_annot.get(cv2.CAP_PROP_FRAME_COUNT):
        print("Different video length encountered! Continuing...")
        print("DEBUG: orig frames: %i, annot frames: %i" % (cap_orig.get(cv2.CAP_PROP_FRAME_COUNT), cap_annot.get(cv2.CAP_PROP_FRAME_COUNT)))
        continue
    
    # Open VideoWriter Objects
    fourcc=cv2.VideoWriter_fourcc(*'FFV1')
    fps=20
    framesize=(640,480)
    isColor=True
    # split the ./recordings/00000_orig.avi into 2 components (head, tail)
    recordings_dir_path, filename_orig = os.path.split(orig_raw_list[i])
    # further split the ./recordings dir to find the project root
    project_root, _ = os.path.split(recordings_dir_path)
    filename_orig, _ = os.path.splitext(filename_orig)
    filename_orig = filename_orig + '_pp.avi'
    if os.path.exists(filename_orig):   # If file exists...
        os.remove(filename_orig)    # ...delete it
    vWriter_orig = cv2.VideoWriter(os.path.join(os.getcwd(), "data", "train", filename_orig), fourcc, fps, framesize, isColor)
    vWriter_orig2 = cv2.VideoWriter(os.path.join(os.getcwd(), "data", "test", filename_orig), fourcc, fps, framesize, isColor)
    
    isColor=False
    _, filename_annot = os.path.split(annot_raw_list[i])
    filename_annot, _ = os.path.splitext(filename_annot)
    filename_annot = filename_annot + '_pp.avi'
    if os.path.exists(filename_annot):  # If file exists...
        os.remove(filename_annot)   # ...delete it
    vWriter_annot = cv2.VideoWriter(os.path.join(os.getcwd(), "data", "validation", filename_annot), fourcc, fps, framesize, isColor)
    
    if not vWriter_orig.isOpened() or not vWriter_annot.isOpened():
        print("Could not open vide writers! Continuing...")
        vWriter_annot.release()
        vWriter_orig.release()
        continue
    
    # Produce output videos
    print("Processing recording nr. {}...".format(i))
    frame_cnt = 0
    while cap_orig.isOpened() and cap_annot.isOpened(): # Iterate through every frame
        ret_o, frame_o = cap_orig.read()
        ret_a, frame_a = cap_annot.read()
        if not ret_o or not ret_a:
            break

        if frame_cnt < int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) * ratio):
            # Postprocess original recording: convert from BGR to RGB
            vWriter_orig.write(cv2.cvtColor(frame_o,cv2.COLOR_BGR2RGB))
            
            # Postprocess annotated frame: binarize it
            annot_binary = binarize(frame_o, frame_a)
            vWriter_annot.write(annot_binary)
        else:
            # write out the test data
            vWriter_orig2.write(cv2.cvtColor(frame_o,cv2.COLOR_BGR2RGB))
        
        frame_cnt += 1
    
    print("Processing of recording nr. {} done.".format(i))
    
    # Release writer resources
    vWriter_annot.release()
    vWriter_orig.release()

print("Post-processing finished!")