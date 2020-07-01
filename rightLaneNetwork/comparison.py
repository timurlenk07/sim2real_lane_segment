import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from dataManagement.myTransforms import testTransform
from models.FCDenseNet.tiramisu import FCDenseNet57


def main(*, dataPath, showCount, baselinePath, cycleganPath, mmePath, **kwargs):
    img_paths = glob.glob(os.path.join(dataPath, '*.png'))
    img_paths = random.sample(img_paths, showCount)

    models = [FCDenseNet57(2, 1), FCDenseNet57(2, 3), FCDenseNet57(2, 1)]

    models[0].load_state_dict(torch.load(baselinePath))
    models[1].load_state_dict(torch.load(cycleganPath))
    models[2].load_state_dict(torch.load(mmePath))
    for i in range(len(models)):
        models[i].eval()

    finalResult = np.empty([0, 4 * 160, 3], dtype=np.uint8)
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

    cv2.imwrite('results/comparison.png', finalResult)
    print("results/comparison.png created.")


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('--showCount', type=int, default=4)
    parser.add_argument('--dataPath', type=str, default='./data/input')
    parser.add_argument('--baselinePath', type=str, default='./FCDenseNet57weights.pth')
    parser.add_argument('--cycleganPath', type=str, default='./FCDenseNet57GANweights.pth')
    parser.add_argument('--mmePath', type=str, default='./FCDenseNet57MMEweights.pth')
    args = parser.parse_args()
    print(vars(args))

    main(**vars(args))
