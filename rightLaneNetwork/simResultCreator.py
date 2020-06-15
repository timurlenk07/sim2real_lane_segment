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
    parser.add_argument('--simPath', type=str, default='./data/input')
    parser.add_argument('--realPath', type=str, default='./data/input')
    parser.add_argument('--weights_path', type=str, default='./results/FCDenseNet57weights.ckpt')
    args = parser.parse_args()

    sim_img_paths = glob.glob(os.path.join(args.simPath, '*.png'))
    random.shuffle(sim_img_paths)
    sim_img_paths = sim_img_paths[:args.showCount]

    real_img_paths = glob.glob(os.path.join(args.realPath, '*.png'))
    random.shuffle(real_img_paths)
    real_img_paths = real_img_paths[:args.showCount]

    model = FCDenseNet57(2, 3)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    transform = myTransformation

    finalResult = np.empty([0, 640, 3], dtype=np.uint8)
    for sim_img_path, real_img_path in zip(sim_img_paths, real_img_paths):
        sim_img = cv2.imread(sim_img_path, cv2.IMREAD_COLOR)
        sim_img = cv2.resize(sim_img, (160, 120), cv2.INTER_LANCZOS4)
        real_img = cv2.imread(real_img_path, cv2.IMREAD_COLOR)
        real_img = cv2.resize(real_img, (160, 120), cv2.INTER_LANCZOS4)

        img_batch = torch.stack([transform(img_, None)[0] for img_ in [sim_img, real_img]])

        _, pred = torch.max(model.forward(img_batch), 1)
        pred = pred.byte()
        pred = [pred_.squeeze().numpy() for pred_ in pred]

        sim_img2 = sim_img.copy()
        sim_img2[pred[0] > 0.5] = (0, 0, 255)
        real_img2 = real_img.copy()
        real_img2[pred[1] > 0.5] = (0, 0, 255)

        result = np.concatenate((sim_img, sim_img2, real_img, real_img2), axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('results/exampleResult.png', finalResult)
