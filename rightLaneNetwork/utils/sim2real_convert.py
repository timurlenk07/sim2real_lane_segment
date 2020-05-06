import glob
import logging
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn


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


def main(dataPath, weightsPath):
    haveCuda = torch.cuda.is_available()
    haveCuda = False

    # Make model
    model = GeneratorResNet([3, 640, 480], 9)
    model.load_state_dict(torch.load(weightsPath))
    model.eval()
    if haveCuda:
        model.cuda()

    # Get list of images in dataPath
    imgs = sorted(glob.glob(os.path.join(dataPath, '**', 'input', '*.png')))
    logging.debug(f"Found images length: {len(imgs)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Process every batch of data
    # TODO use batches
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for img_p in imgs:
        img = cv2.imread(img_p)
        cv2.imwrite('test_in.png', img)
        img = transform(img).unsqueeze(0)
        if haveCuda:
            img.cuda()

        img = model.forward(img)
        img = img.squeeze().detach().numpy()
        img = (img + 1) / 2 * 255
        img = img.transpose([1, 2, 0]).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_out.png', img)
        break
        # cv2.imwrite(img_p, img)


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='./data')
    parser.add_argument('--modelWeightsPath', type=str, default='./G_BA_0.pth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    main(args.dataPath, args.modelWeightsPath)
