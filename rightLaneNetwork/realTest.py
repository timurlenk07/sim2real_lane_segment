import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from RightLaneMMEModule import RightLaneMMEModule
from RightLaneModule import RightLaneModule
from dataManagement.myDatasets import RightLaneDataset


def main(*, module_type, checkpoint_path, showCount, realDataPath, simDataPath, testDataPath, **kwargs):
    # Parse model
    if module_type == 'MME':
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
    elif module_type in ['baseline', 'CycleGAN']:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
    else:
        raise RuntimeError(f"Cannot recognize module type {args.module_type}")

    model.eval()
    print(f"Loaded {model.__class__} instance.")

    # Get transformation from model
    transform = model.transform

    # Get real images from folder
    img_paths = glob.glob(os.path.join(realDataPath, '*.png'))

    # Randomly sample showCount number of images
    img_paths = random.sample(img_paths, showCount)
    # img_paths = img_paths[:showCount]

    # Create samples from real image predictions
    finalResult = np.empty([0, 2 * model.width, 3], dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (model.width, model.height), cv2.INTER_LANCZOS4)

        img_prep, _ = transform(img, None)
        img_prep = img_prep.unsqueeze(0)

        _, pred = torch.max(model.forward(img_prep), 1)
        pred = pred.byte().numpy().squeeze()

        img2 = img.copy()
        img2[pred > 0.5] = (0, 0, 255)

        result = np.concatenate((img, img2), axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('results/predsReal.png', finalResult)

    # Perform qualitative evaluation
    testDataset = RightLaneDataset(testDataPath, transform=transform)

    correct = 0
    total = 0
    for i in range(len(testDataset)):
        img, label = testDataset[i]
        img = img.unsqueeze(0)
        _, pred = torch.max(model.forward(img), 1)
        pred = pred.squeeze()
        correct += pred.eq(label).sum().item()
        total += label.numel()

    print(f"Accuracy on test set: {correct * 100.0 / (total + 1e-6):.2f}%")


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('-t', '--module_type', required=True, choices=['baseline', 'CycleGAN', 'MME'])
    parser.add_argument('--checkpoint_path', type=str, default='./results/FCDenseNet57.ckpt')
    parser.add_argument('-c', '--showCount', type=int, default=5)
    parser.add_argument('--realDataPath', type=str, default='./data/input')
    parser.add_argument('--simDataPath', type=str, default='./data/input')
    parser.add_argument('--testDataPath', type=str, default='./data')
    args = parser.parse_args()
    print(vars(args))

    main(**vars(args))
