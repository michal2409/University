import argparse
import random
import shutil
import imageio
import numpy as np
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
from util import check_array, create_dir_if_not_exist
from numba import uint16, jit

TRAIN_SPLIT = 0.8
IMG_SIZE = 256
SEED = 1

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='cityscapes', help="Directory with the cityscapes dataset")
parser.add_argument('--output_dir', default='data', help="Where to write the new data")

def convert_to_label(img):
    """Converts img pixels to colors class based on check_array."""
    labeled = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint16)
    for row in range(IMG_SIZE):
        for col in range(IMG_SIZE):
            for i in range(check_array.shape[0]):
                if (img[row, col, :] == check_array[i]).all():
                    labeled[row][col] = i
                    break
    return labeled

convert_to_label_fast = jit(uint16[:,:](uint16[:,:,:]))(convert_to_label)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # storing filenames from data_dir with *.png pattern
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.png')]

    # shuffling filneames
    random.seed(SEED)
    filenames.sort()
    random.shuffle(filenames)

    # splitting into training and validation set
    split = int(TRAIN_SPLIT * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]
    splited_sets = {'train': train_filenames, 'val': val_filenames}

    create_dir_if_not_exist(args.output_dir)

    for split in ['train', 'val']:
        """Saving to directories:
            X - original images
            Y - segmented images
            labeled - images with converted colors from segmented images to classes
        """
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        output_dir_labeled = os.path.join(output_dir_split, 'labeled')
        output_dir_X = os.path.join(output_dir_split, 'X')
        output_dir_Y = os.path.join(output_dir_split, 'Y')
        create_dir_if_not_exist(output_dir_split)
        create_dir_if_not_exist(output_dir_labeled)
        create_dir_if_not_exist(output_dir_X)
        create_dir_if_not_exist(output_dir_Y)

        # saving image, segmented image and labeled image
        for filename in tqdm(splited_sets[split]):
            img = plt.imread(filename) * 255
            X, Y = img[:, :IMG_SIZE, :], img[:, IMG_SIZE:, :]
            labeled = convert_to_label_fast(Y.astype(np.uint8))

            fname = os.path.basename(os.path.normpath(filename))
            imageio.imwrite(os.path.join(output_dir_X, fname), X.astype(np.uint8))
            imageio.imwrite(os.path.join(output_dir_Y, fname), Y.astype(np.uint8))
            imageio.imwrite(os.path.join(output_dir_labeled, fname), labeled.astype(np.uint8))
