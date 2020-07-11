import argparse
import random
from concurrent.futures import ThreadPoolExecutor

from myDatasets import RightLaneDataset
from skimage.exposure import match_histograms
from skimage.util import img_as_float, img_as_ubyte
from tqdm import tqdm


def main(ds_source, ds_reference, no_shuffle, workers):
    if workers <= 0:
        workers = 1
    ds_source = RightLaneDataset(ds_source, transform=lambda x, y: (x, y), haveLabels=False)
    ds_reference = RightLaneDataset(ds_reference, transform=lambda x, y: (x, y), haveLabels=False)

    # Create index lists, shuffle if required
    idxes = list(range(len(ds_source)))
    ref_idxes = list(range(len(ds_reference)))
    if not no_shuffle:
        random.shuffle(ref_idxes)

    # Histogram matching of a single index
    def processIdx(idx):
        ref_idx = ref_idxes[idx % len(ref_idxes)]
        source_img, _ = ds_source[idx]
        reference_img, _ = ds_reference[ref_idx]
        matched_img = match_histograms(img_as_float(source_img), img_as_float(reference_img), multichannel=True)
        ds_source[idx] = img_as_ubyte(matched_img)

    # Use the power of concurrency to speed up things
    print(f"Matching histograms using {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(processIdx, idxes), total=len(idxes)))

    print("Finished matching histograms.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_source', type=str, help="Dataset wanted to be changed.")
    parser.add_argument('--ds_reference', type=str, help="Dataset of matching reference.")
    parser.add_argument('--no_shuffle', action='store_false', help="Whether to skip shuffling images before matching.")
    parser.add_argument('--workers', type=int, default=4, help="Number of CPU workers to speed up parallel processing.")
    args = parser.parse_args()

    main(**vars(args))
