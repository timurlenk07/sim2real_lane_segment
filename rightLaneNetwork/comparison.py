import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from RightLaneModule import myTransformation
from models.FCDenseNet.tiramisu import FCDenseNet57

if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('--showCount', type=int, default=4)
    parser.add_argument('--realPath', type=str, default='./data/input')
    parser.add_argument('--baseline_path', type=str, default='./FCDenseNet57weights.pth')
    parser.add_argument('--cyclegan_path', type=str, default='./FCDenseNet57GANweights.pth')
    parser.add_argument('--mme_path', type=str, default='./FCDenseNet57MMEweights.pth')
    args = parser.parse_args()

    img_paths = glob.glob(os.path.join(args.realPath, '*.png'))
    random.shuffle(img_paths)
    img_paths = img_paths[:args.showCount]

    models = [FCDenseNet57(2, 1), FCDenseNet57(2, 3), FCDenseNet57(2, 1)]

    models[0].load_state_dict(torch.load(args.baseline_path))
    models[1].load_state_dict(torch.load(args.cyclegan_path))
    models[2].load_state_dict(torch.load(args.mme_path))
    for i in range(len(models)):
        models[i].eval()

    transform = myTransformation

    finalResult = np.empty([0, 640, 3], dtype=np.uint8)
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img_batch = [transform(img, None), transform(img, None), transform(img, None)]
        img_batch = [img_[0].unsqueeze(0) for img_ in img_batch]

        pred_imgs = []
        for model, img_ in zip(models, img_batch):
            model.eval()
            _, pred = torch.max(model.forward(img_), 1)
            pred = pred.byte().numpy()
            pred_imgs.append(pred[0].squeeze() > 0.5)

        img = cv2.resize(img, (160, 120), cv2.INTER_LANCZOS4)
        imgs = [img]
        # pred_imgs[2] = np.transpose(pred_imgs[2])
        for pred in pred_imgs:
            img2 = img.copy()
            img2[pred] = (0, 0, 255)
            imgs.append(img2)

        result = np.concatenate(imgs, axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('results/exampleResult.png', finalResult)