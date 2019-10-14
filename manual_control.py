#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from skimage.io import imsave, imread
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
from threading import Thread
from gym_duckietown.recorder import Recorder

# from experiments.utils import save_img

pyglet.options['debug_gl'] = False
pyglet.options['vsync'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-loop-v0')
parser.add_argument('--map-name', default='loop')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--annotated', default=0, action='store_const', const=0, help='start the simulation with annotated texture segments')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
        annotated = args.annotated,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

recordingInProgress = False
recorder_orig = Recorder()
recorder_annot = Recorder()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    global recordingInProgress
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    elif symbol == key.A:
        print('Annotation mode set to: {}'.format(env.annotated+1))
        env.annotated = (env.annotated + 1) % 3

    # Start/Stop video recording
    elif symbol == key.RETURN:
        if recordingInProgress:
            print('stop recording')
            t_orig = Thread(target=recorder_orig.save, kwargs={"filename":"orig"})
            t_annot = Thread(target=recorder_annot.save, kwargs={"filename":"annot"})
            t_orig.start()
            t_annot.start()
            env.recording_time = 0.0
        else:
            print('start recording')
        recordingInProgress = not recordingInProgress
        


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    global recordingInProgress, hasVideoStarted, video_orig, video_annot

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    # if key_handler[key.RETURN]:
    #     from PIL import Image
    #     im = Image.fromarray(obs)
    #
    #     im.save('screen.png')

    # record the original and annotated frames
    if recordingInProgress:
        if env.annotated == 0:
            # cannot start form unannotated mode
            # the program doesn't know which annotation mode to pick for the recording
            print("Stopping recording. Cannot start in unannotated mode.")
            recordingInProgress = False
        else:
            # get the original and annotated image data
            # using the observer object/methods instead of the env.render("rgb_array"), this
            # results in -1 rendering calls (also a 640x480 frame instead of 800x600 one)
            img_annot = obs
            annotated_state = env.annotated
            env.annotated = 0
            img_orig = env.render_obs()
            env.annotated = annotated_state

            # save the data in a buffer
            recorder_orig.record(img_orig)
            recorder_annot.record(img_annot)

            # limit the max recording length to 10 sec
            env.recording_time += dt    # add the delta time [s] to the recording time
            if env.recording_time > 10.0:
                recordingInProgress = False
                t_orig = Thread(target=recorder_orig.save, kwargs={"filename": "orig"})
                t_annot = Thread(target=recorder_annot.save, kwargs={"filename": "annot"})
                t_orig.start()
                t_annot.start()
                env.recording_time = 0.0

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

if hasVideoStarted:
    video_orig.release()
    video_annot.release()

env.close()
