import functools
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
from dataManagement.myTransforms import testTransform


def main(*, module_type, checkpointPath, showCount, realDataPath, trainDataPath, testDataPath, **kwargs):
    # Parse model
    if module_type == 'MME':
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=checkpointPath)
    elif module_type in ['baseline', 'CycleGAN']:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=checkpointPath)
    else:
        raise RuntimeError(f"Cannot recognize module type {module_type}")

    model.eval()
    print(f"Loaded {model.__class__} instance.")

    # Get transform function
    transform = functools.partial(testTransform, width=model.width, height=model.height, gray=model.grayscale)

    # Randomly sample showCount number of images from training and real folders
    train_img_paths = glob.glob(os.path.join(trainDataPath, '*.png'))
    train_img_paths = random.sample(train_img_paths, showCount)
    real_img_paths = glob.glob(os.path.join(realDataPath, '*.png'))
    real_img_paths = random.sample(real_img_paths, showCount)

    # Create samples from training and real image predictions
    finalResult = np.empty([0, 4 * model.width, 3], dtype=np.uint8)
    for train_img_path, real_img_path in zip(train_img_paths, real_img_paths):
        train_img = cv2.imread(train_img_path, cv2.IMREAD_COLOR)
        train_img = cv2.resize(train_img, (model.width, model.height), cv2.INTER_LANCZOS4)
        real_img = cv2.imread(real_img_path, cv2.IMREAD_COLOR)
        real_img = cv2.resize(real_img, (model.width, model.height), cv2.INTER_LANCZOS4)

        img_batch = [train_img, real_img]
        img_batch = torch.stack([transform(img_)[0] for img_ in img_batch])

        _, pred = torch.max(model.forward(img_batch), 1)
        pred = pred.byte()
        pred = [pred_.squeeze().numpy() for pred_ in pred]

        train_img2 = train_img.copy()
        train_img2[pred[0] > 0.5] = (0, 0, 255)
        real_img2 = real_img.copy()
        real_img2[pred[1] > 0.5] = (0, 0, 255)

        result = np.concatenate((train_img, train_img2, real_img, real_img2), axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('results/samplePredictions.png', finalResult)

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
    parser.add_argument('--checkpointPath', type=str, default='./results/FCDenseNet57.ckpt')
    parser.add_argument('-c', '--showCount', type=int, default=5)
    parser.add_argument('--realDataPath', type=str, default='./data/input')
    parser.add_argument('--trainDataPath', type=str, default='./data/input')
    parser.add_argument('--testDataPath', type=str, default='./data')
    args = parser.parse_args()
    print(vars(args))

    main(**vars(args))