import glob
import json
import os
import os.path as osp
import shutil

import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm

label_name_to_value = {
    "_background_": 0,
    "right_lane": 1,
    "left_lane": 2,
    "obstacle": 3,
}

imitateBehaviour = False


def findLabelledImgs(dataPath, labelPath):
    imgs_p = sorted(glob.glob(osp.join(dataPath, '*.png')), reverse=True)
    labels_p = sorted(glob.glob(osp.join(labelPath, '*.json')), reverse=True)

    labelledImgs_p = [osp.basename(label_p).split('.json')[0] for label_p in labels_p]
    labelledImgs_p = [osp.join(dataPath, p + '.png') for p in labelledImgs_p]
    unlabelledImgs_p = [p for p in imgs_p if p not in labelledImgs_p]

    labelledPairs = [{'image': img, 'label': label}
                     for img, label in zip(labelledImgs_p, labels_p)]

    return labelledPairs, unlabelledImgs_p


def createUnlabelledDB(imgs_p, unlabelled_dir):
    if not imitateBehaviour:
        os.makedirs(unlabelled_dir, exist_ok=True)

    imgs_p.sort()
    for i, img_p in tqdm(enumerate(imgs_p)):
        if not imitateBehaviour:
            shutil.move(img_p, osp.join(unlabelled_dir, f'{i:06d}.png'))


def createLabelledDB(labelledPairs, input_dir, label_dir):
    if not imitateBehaviour:
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

    for i, labelledPair in tqdm(enumerate(labelledPairs)):
        label = json.load(open(labelledPair['label']))  # Load JSON
        img = cv2.imread(labelledPair['image'])  # Load image to get its shape later

        for shape in sorted(label["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            assert label_name in label_name_to_value, f"Got unknown label: {label_name}"
        lbl, _ = utils.shapes_to_label(img.shape, label["shapes"], label_name_to_value)

        if not imitateBehaviour:
            shutil.move(labelledPair['image'], osp.join(input_dir, f'{i:06d}.png'))
            cv2.imwrite(osp.join(label_dir, f'{i:06d}.png'), lbl.astype(np.uint8))


# Assert data is under dataPath directory
# Assert labels are under dataPath/annotations directory
def main(dataPath, deleteLabels=False):
    labelPath = osp.join(dataPath, 'annotations')
    assert osp.isdir(dataPath), f"Given data path must point to a directory! Got: {dataPath}"
    assert osp.isdir(labelPath), f"Label directory not found under: {labelPath}"

    # Sort data into labelled and unlabelled categories
    labelledPairs, unlabelledImgs_p = findLabelledImgs(dataPath, labelPath)
    print(f"Found {len(labelledPairs)} labelled samples and {len(unlabelledImgs_p)} unlabelled samples.")

    # Create unlabelled DB
    createUnlabelledDB(unlabelledImgs_p, unlabelled_dir=osp.join(dataPath, 'unlabelled'))

    # Create labelled DB
    createLabelledDB(labelledPairs, input_dir=osp.join(dataPath, 'input'), label_dir=osp.join(dataPath, 'label'))

    print(f"Finished creating real database.")

    if deleteLabels:
        if not imitateBehaviour:
            shutil.rmtree(labelPath)
        print(f"Removed folder containing JSON labels.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, required=True,
                        help="Root of the data directory containing the real images.")
    parser.add_argument('--deleteLabels', action='store_true',
                        help="Delete folder containing JSON labels after processing is done.")
    parser.add_argument('--imitate', action='store_true',
                        help="Imitate behaviour without modifying anything. For development purposes.")
    args = parser.parse_args()

    if args.imitate:
        imitateBehaviour = True
    del args.imitate

    main(**vars(args))
