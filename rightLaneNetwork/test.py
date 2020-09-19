import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import accuracy, dice_score, iou, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from RightLaneMMEModule import RightLaneMMEModule
from RightLaneModule import RightLaneModule
from RightLaneSTModule import RightLaneSTModule
from dataManagement.myDatasets import RightLaneDataset
from dataManagement.myTransforms import MyTransform


def main(*, module_type, checkpointPath, showCount, realDataPath, trainDataPath, testDataPath):
    # Ensure reproducibility
    seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Parse model
    if module_type == 'mme':
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=checkpointPath)
    elif module_type in ['baseline', 'hm', 'CycleGAN']:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=checkpointPath)
    elif module_type == 'sandt':
        model = RightLaneSTModule.load_from_checkpoint(checkpoint_path=checkpointPath)
    else:
        raise RuntimeError(f"Cannot recognize module type {module_type}")

    model.eval()
    print(f"Loaded {model.__class__} instance.")
    print(f"Model has the following hyperparameters: {model.hparams}")

    # Get transform function
    testTransform = MyTransform(width=model.width, height=model.height, gray=model.grayscale, augment=False)

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
        img_batch = torch.stack([testTransform(img_)[0] for img_ in img_batch])

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
    testDataset = RightLaneDataset(testDataPath, transform=testTransform, haveLabels=True)
    testDataLoader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=8)

    if torch.cuda.is_available():
        model = model.cuda()

    test_acc, test_dice, test_iou = 0.0, 0.0, 0.0
    test_conf_matrix = torch.zeros(2, 2, device=model.device)
    totalWeight = 0
    for batch in tqdm(testDataLoader):
        img, label = batch
        if torch.cuda.is_available():
            img, label = img.cuda(), label.cuda()

        probas = model.forward(img)
        _, label_hat = torch.max(probas, 1)

        weight = img.shape[0]
        test_acc += accuracy(label_hat, label, num_classes=2) * weight
        test_dice += dice_score(probas, label) * weight
        test_iou += iou(label_hat, label, num_classes=2) * weight
        test_conf_matrix += confusion_matrix(label_hat, label)
        totalWeight += weight

    assert totalWeight == len(testDataset)

    if len(testDataset) > 0:
        test_acc /= len(testDataset)
        test_dice /= len(testDataset)
        test_iou /= len(testDataset)

    print(f"Accuracy on test set: {test_acc * 100.0:.4f}%")
    print(f"Dice score on test set: {test_dice:.4f}")
    print(f"IoU on test set: {test_iou * 100.0:.4f}")
    print(f"Confusion matrix (column: prediction, row: label):")
    print(test_conf_matrix)
    print(f"Total: {torch.sum(test_conf_matrix)}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-t', '--module_type', required=True, choices=['baseline', 'sandt', 'hm', 'CycleGAN', 'mme'])
    parser.add_argument('--checkpointPath', type=str)
    parser.add_argument('-c', '--showCount', type=int, default=5)
    parser.add_argument('--realDataPath', type=str)
    parser.add_argument('--trainDataPath', type=str)
    parser.add_argument('--testDataPath', type=str)
    args = parser.parse_args()

    main(**vars(args))
