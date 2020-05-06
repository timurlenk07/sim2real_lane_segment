import os
from argparse import ArgumentParser

import cv2
import torch

from RightLaneMMEModule import RightLaneMMEModule
from RightLaneModule import RightLaneModule, myTransformation


def main(inputVideo, outputVideo, model, transform, args=None):
    cap_in = cv2.VideoCapture(inputVideo)

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    fps = 25
    framesize = (160, 120)
    isColor = True
    w_out = cv2.VideoWriter(outputVideo, fourcc, fps, framesize, isColor)

    while cap_in.isOpened() and w_out.isOpened():
        ret, frame = cap_in.read()
        if not ret:
            break

        frame_in, _ = transform(frame, None)
        frame_in = frame_in.unsqueeze(0)

        _, pred = torch.max(model.forward(frame_in), 1)
        pred = pred.byte().numpy().squeeze()
        if args.test_mme:
            pred = pred.transpose(-1, -2)  # Trained with PIL transforms as preprocess

        frame_out = cv2.resize(frame, framesize, cv2.INTER_LANCZOS4)
        frame_out[pred > 0.5] = (0, 0, 255)

        w_out.write(frame_out)

    cap_in.release()
    w_out.release()


if __name__ == '__main__':
    assert torch.cuda.device_count() <= 1

    parser = ArgumentParser()

    parser.add_argument('--test_mme', action='store_true')
    parser.add_argument('--modelCheckpoint', type=str, default='./results/FCDenseNet57.ckpt')
    parser.add_argument('--videoPath', type=str, default='./testVideo.avi')
    parser.add_argument('--outputPath', type=str, default='./demoVideo.avi')
    args = parser.parse_args()

    if os.path.exists(args.outputPath):
        os.remove(args.outputPath)

    if args.test_mme:
        model = RightLaneMMEModule.load_from_checkpoint(checkpoint_path=args.modelCheckpoint)
    else:
        model = RightLaneModule.load_from_checkpoint(checkpoint_path=args.modelCheckpoint)

    transformation = myTransformation if not args.test_mme else model.transform

    model.eval()
    main(args.videoPath, args.outputPath, model, transformation, args)
