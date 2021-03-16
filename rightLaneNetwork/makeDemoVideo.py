import os
from argparse import ArgumentParser

import cv2
import torch
from tqdm import trange

from trainingModules.SimpleTrain import SimpleTrainModule
from trainingModules.MMETrainingModule import MMETrainingModule
from dataManagement.myTransforms import MyTransform

haveCuda = torch.cuda.is_available()


def predictVideo(inputVideo, outputVideo, model, transform):
    cap_in = cv2.VideoCapture(inputVideo)

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    framesize = (160, 120)
    isColor = True
    w_out = cv2.VideoWriter(outputVideo, fourcc, fps, framesize, isColor)

    try:
        for _ in trange(int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap_in.read()
            if not ret:
                continue

            frame_in, _ = transform(frame)
            frame_in = frame_in.unsqueeze(0)

            if haveCuda:
                frame_in = frame_in.cuda()

            _, pred = torch.max(model.forward(frame_in), 1)
            pred = pred.cpu().byte().numpy().squeeze()

            frame_out = cv2.resize(frame, framesize, cv2.INTER_LANCZOS4)
            frame_out[pred == 1] = (0, 255, 0)  # Right lane
            frame_out[pred == 2] = (255, 0, 0)  # Left lane
            frame_out[pred == 3] = (0, 0, 255)  # Obstacles

            w_out.write(frame_out)
    finally:
        cap_in.release()
        w_out.release()


def main(videoIns, videoOuts, module_type, ckpt_path):
    # Parse model
    if module_type == 'MME':
        model = MMETrainingModule.load_from_checkpoint(checkpoint_path=ckpt_path, num_cls=4)
    elif module_type in ['baseline', 'sandt', 'hm', 'CycleGAN']:
        model = SimpleTrainModule.load_from_checkpoint(checkpoint_path=ckpt_path, num_cls=4)
    else:
        raise RuntimeError(f"Cannot recognize module type {module_type}")

    if haveCuda:
        model.cuda()
    model.eval()

    # Get transform function
    transform = MyTransform(augment=False)

    for videoIn, videoOut in zip(videoIns, videoOuts):
        if os.path.exists(videoOut):
            os.remove(videoOut)

        predictVideo(videoIn, videoOut, model, transform)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-t', '--module_type', required=True, choices=['baseline', 'sandt', 'hm', 'CycleGAN', 'MME'])
    parser.add_argument('--checkpointPath', type=str)
    parser.add_argument('--videoIns', type=str, nargs='+')
    parser.add_argument('--videoOuts', type=str, default='./demoVideo.avi', nargs='+')
    args = parser.parse_args()

    assert len(args.videoIns) == len(args.videoOuts)

    main(args.videoIns, args.videoOuts, args.module_type, args.checkpointPath)
