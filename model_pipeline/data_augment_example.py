#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import os
import sys
import glob
import argparse
from collections import defaultdict
from PIL import Image
import random


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--train', default= 1000 , type=int, help='target size for training')
    parser.add_argument('--valid', default= 200 , type=int, help='target size for vaidation')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='reflect')

    img = load_img('../data/images_jpeg/102.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='.', save_prefix='aug', save_format='jpeg'):
        i += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely




if __name__ == "__main__":
  main()
