import glob
import logging
import math
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tqdm import tqdm


# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def main(dataPath, overwriteData, weightsPath, batch_size):
    haveCuda = torch.cuda.is_available()

    # Make model
    model = GeneratorResNet((3, 120, 160), 9)
    model.load_state_dict(torch.load(weightsPath))
    model.eval()
    if haveCuda:
        model = model.cuda()

    # Get list of images in dataPath
    imgs = sorted(glob.glob(os.path.join(dataPath, '**', 'input', '*.png'), recursive=True))
    print(f"Found images length: {len(imgs)}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((120, 160), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Process every batch of data
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for img_p_batch in tqdm(batch(imgs, batch_size), total=math.ceil(len(imgs) / batch_size)):
        img_batch = []
        for img_p in img_p_batch:
            img = cv2.imread(img_p)
            img = transform(img)
            img_batch.append(img)

        img_batch = torch.from_numpy(np.stack(img_batch))
        if haveCuda:
            img_batch = img_batch.cuda()
        img_batch = model(img_batch)

        for i, img_p in enumerate(img_p_batch):
            img = img_batch[i].detach().cpu().squeeze().numpy()
            img = (img.transpose([1, 2, 0]) + 1) / 2
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (640, 480), cv2.INTER_LANCZOS4)
            cv2.imwrite(img_p, img)


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()
    parser.add_argument('--dataPath', type=str)
    parser.add_argument('--overwriteData', action='store_true', help="Currently unused.")
    parser.add_argument('--modelWeightsPath', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.dataPath, args.overwriteData, args.modelWeightsPath, args.batch_size)
