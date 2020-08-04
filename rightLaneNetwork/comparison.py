import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from dataManagement.myTransforms import testTransform
from models.FCDenseNet.tiramisu import FCDenseNet57


def main(*, dataPath, showCount, baselinePath, sandtPath, hmPath, cycleganPath, mmePath, resultPath):
    img_paths = glob.glob(os.path.join(dataPath, '*.png'))
    img_paths = random.sample(img_paths, showCount)

    models = [FCDenseNet57(n_classes=2) for _ in range(5)]

    models[0].load_state_dict(torch.load(baselinePath))
    models[1].load_state_dict(torch.load(sandtPath))
    models[2].load_state_dict(torch.load(hmPath))
    models[3].load_state_dict(torch.load(cycleganPath))
    models[4].load_state_dict(torch.load(mmePath))
    for i in range(len(models)):
        models[i].eval()

    finalResult = np.empty([0, 6 * 160, 3], dtype=np.uint8)
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img_batch = [testTransform(img, width=160, height=120, gray=False)[0].unsqueeze(0) for _ in models]

        pred_imgs = []
        for model, img_ in zip(models, img_batch):
            model.eval()
            _, pred = torch.max(model.forward(img_), 1)
            pred = pred.byte().numpy()
            pred_imgs.append(pred[0].squeeze() > 0.5)

        img = cv2.resize(img, (160, 120), cv2.INTER_LANCZOS4)
        imgs = [img]

        for pred in pred_imgs:
            img2 = img.copy()
            img2[pred] = (0, 0, 255)
            imgs.append(img2)

        result = np.concatenate(imgs, axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite(resultPath, finalResult)
    print(f"{resultPath} created.")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--showCount', type=int, default=4)
    parser.add_argument('--dataPath', type=str)
    parser.add_argument('--baselinePath', type=str)
    parser.add_argument('--sandtPath', type=str)
    parser.add_argument('--hmPath', type=str)
    parser.add_argument('--cycleganPath', type=str)
    parser.add_argument('--mmePath', type=str)
    parser.add_argument('--resultPath', type=str, default='results/comparison.png')
    args = parser.parse_args()
    print(vars(args))

    main(**vars(args))
