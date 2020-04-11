import argparse
import glob
import os.path
import random

import numpy as np
import skvideo.io
from tqdm import tqdm


# Return 'num_images' no. of random images
def random_items(iterator, num_images=1):
    selected_items = [None] * num_images

    for item_index, item in enumerate(iterator):
        for selected_item_index in range(num_images):
            if not random.randint(0, item_index):
                selected_items[selected_item_index] = item

    return selected_items


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', default=100)
    parser.add_argument('--save_path', default="train")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    dir = '..'
    files = glob.glob(os.path.join(dir, '*.mp4'))
    videoset = [None]

    print("Reading and joining videos")
    with tqdm(total=len(files)) as bar:
        for f in range(len(files)):
            videogen = skvideo.io.vreader(files[f])
            for item_index, item in enumerate(videogen):
                videoset.append(item)

            bar.update(1)
    print("Total number of frames = ", len(videoset))

    print("Generating random samples and saving as .npy files")
    with tqdm(total=int(args.num_images)) as bar1:

        for i in range(300):
            random_set = random_items(videoset, int(int(args.num_images) / 300))

            count = 0

            for frame in random_set:
                directory = "{0}/dt_real_sample_{1}_{2}".format(args.save_path, i, count)
                np.save(directory,
                        np.array(frame, dtype=np.float32))
                count += 1
                bar1.update(1)
