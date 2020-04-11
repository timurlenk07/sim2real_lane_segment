import glob
import random

import cv2
import numpy as np
import torch

from old.rightLaneSegment import MyTransform, EncDecNet

assert torch.cuda.device_count() <= 1

showCount = 5

img_paths = sorted(glob.glob('train/*.npy'))
random.shuffle(img_paths)
img_paths = img_paths[:showCount]

transform = MyTransform(grayscale=False)
net = EncDecNet(16, 3, 5, 'leakyRelu', inFeat=3)
net.load_state_dict(torch.load('./results/EncDecNet.pth'))
net.eval()

finalResult = np.empty([0, 320, 3], dtype=np.uint8)
for i, img_path in enumerate(img_paths):
    img = np.load(img_path)
    img = cv2.resize(img, (160, 120))

    img_prep, _ = transform(img, None)
    img_prep = img_prep.unsqueeze(0)

    _, pred = torch.max(net(img_prep), 1)
    pred = pred.byte().numpy().squeeze()
    # pred = cv2.resize(pred, (160, 120))

    img2 = img.copy()
    img2[pred > 0.5] = (0, 0, 255)

    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    result = np.concatenate((img, img2), axis=1)
    finalResult = np.concatenate((finalResult, result), axis=0)

cv2.imwrite('./results/predsReal.png', finalResult)
