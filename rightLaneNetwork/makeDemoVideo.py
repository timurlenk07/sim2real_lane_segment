import functools
import os
from argparse import ArgumentParser

import cv2
import torch
from tqdm import trange

from RightLaneMMEModule import RightLaneMMEModule
from RightLaneModule import RightLaneModule
from RightLaneSTModule import RightLaneSTModule
from dataManagement.myTransforms import testTransform


def main(inputVideo, outputVideo, model, transform):
    cap_in = cv2.VideoCapture(inputVideo)

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    framesize = (model.width, model.height)
    isColor = not model.grayscale
    w_out = cv2.VideoWriter(outputVideo, fourcc, fps, framesize, isColor)

    for _ in trange(int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap_in.read()
        if not ret:
            continue

        frame_in, _ = transform(frame)
        frame_in = frame_in.unsqueeze(0)

        _, pred = torch.max(model.forward(frame_in), 1)
        pred = pred.byte().numpy().squeeze()

        frame_out = cv2.resize(frame, framesize, cv2.INTER_LANCZOS4)
        frame_out[pred > 0.5] = (0, 0, 255)

        w_out.write(frame_out)

    cap_in.release()
    w_out.release()


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('-t', '--module_type', required=True, choices=['baseline', 'CycleGAN', 'MME', 'sandt'])
    parser.add_argument('--checkpointPath', type=str, default='./results/FCDenseNet57.ckpt')
    parser.add_argument('--videoPath', type=str, default='./testVideo.avi')
    parser.add_argument('--outputPath', type=str, default='./demoVideo.avi')
    args = parser.parse_args()

    if os.path.exists(args.outputPath):
        os.remove(args.outputPath)

    # Parse model
    if args.module_type == 'MME':
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=args.checkpointPath)
    elif args.module_type in ['baseline', 'CycleGAN']:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=args.checkpointPath)
    elif args.module_type == 'sandt':
        model = RightLaneSTModule.load_from_checkpoint(checkpoint_path=args.checkpointPath)
    else:
        raise RuntimeError(f"Cannot recognize module type {args.module_type}")

    # Get transform function
    transform = functools.partial(testTransform, width=model.width, height=model.height, gray=model.grayscale)

    model.eval()
    main(args.videoPath, args.outputPath, model, transform)
