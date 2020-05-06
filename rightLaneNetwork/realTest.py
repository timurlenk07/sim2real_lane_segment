import glob
import os.path
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from RightLaneMMEModule import RightLaneMMEModule
from RightLaneModule import RightLaneModule, myTransformation

if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('--test_mme', action='store_true')
    parser.add_argument('--showCount', type=int, default=5)
    parser.add_argument('--dataPath', type=str, default='./data')
    parser.add_argument('--checkpoint_path', type=str, default='./results/FCDenseNet57.ckpt')
    args = parser.parse_args()

    img_paths = glob.glob(os.path.join(args.dataPath, '*.png'))
    random.shuffle(img_paths)
    img_paths = img_paths[:args.showCount]

    if args.test_mme:
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    else:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.eval()

    finalResult = np.empty([0, 320, 3], dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (160, 120), cv2.INTER_LANCZOS4)

        if args.test_mme:
            img_prep, _ = model.transform(img, None)
        else:
            img_prep, _ = myTransformation(img, None)
        img_prep = img_prep.unsqueeze(0)

        _, pred = torch.max(model(img_prep), 1)
        pred = pred.byte().numpy().squeeze()
        if args.test_mme:
            pred = pred.transpose(-1, -2)  # Trained with PIL transforms as preprocess
        # pred = cv2.resize(pred, (160, 120))

        img2 = img.copy()
        img2[pred > 0.5] = (0, 0, 255)

        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        result = np.concatenate((img, img2), axis=1)
        finalResult = np.concatenate((finalResult, result), axis=0)

    cv2.imwrite('results/predsReal.png', finalResult)
