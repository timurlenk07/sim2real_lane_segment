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

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-loop-v0')
parser.add_argument('--map-name', default='loop')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--annotated', default=False, action='store_true', help='start the simulation with annotated texture segments')
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

hasVideoStarted = False
video_orig = None
video_annot = None
recordingInProgress = False

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
        if hasVideoStarted:
            video_orig.release()
            video_annot.release()
        sys.exit(0)

    elif symbol == key.A:
        print('ANNOTATING')
        env.annotated = not env.annotated

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    elif symbol == key.RETURN:
        if recordingInProgress:
            print('stop recording')
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
    
    if recordingInProgress:
        annotated_state = env.annotated
        env.annotated = False
        img = env.render('rgb_array')
        if not hasVideoStarted:
            print(img.shape)
            height, width, layers = img.shape
            video_name_orig = 'testvideo_orig.avi'
            video_name_annot = 'testvideo_annot.avi'
            video_orig = cv2.VideoWriter(video_name_orig, cv2.VideoWriter_fourcc(*'FFV1'), 20, (width,height))
            video_annot = cv2.VideoWriter(video_name_annot, cv2.VideoWriter_fourcc(*'FFV1'), 20, (width,height))
            hasVideoStarted = True
        
        #imsave('screenshot.png', np.array(img, dtype=np.uint8))
        video_orig.write(img)
        
        env.annotated = True
        img = env.render('rgb_array')
        video_annot.write(img)
        
        env.annotated = annotated_state
        #if env.annotated:
            #env.annotated = False
            #img = env.render('rgb_array')
            #imsave('screenshot_a.png', np.array(img, dtype=np.uint8))
            #env.annotated = True

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
