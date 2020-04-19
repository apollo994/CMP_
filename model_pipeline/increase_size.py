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


def increase_sample(target_folder, target_size, datagen):
    dirs = [x[0] for x in os.walk(target_folder)]
    dirs = dirs[1:]

    for dir in dirs:

        imgs = glob.glob(dir+'/*.jpg')
        n_file = len(imgs)
        n_to_add = target_size - n_file
        print ()
        print (f'{n_file} images in {dir}')
        print (f'will bring this number to {target_size}')

        for i in range(round(n_to_add/2)):
            rand = random.sample(imgs,1)
            im = Image.open(rand[0])
            im= im.resize((150, 150), Image.ANTIALIAS)
            x = img_to_array(im)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=dir, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i > 1:
                    break

        print (f'{n_to_add} new images added to {dir}')


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--train', default= 1000 , type=int, help='target size for training')
    parser.add_argument('--valid', default= 200 , type=int, help='target size for vaidation')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')

    args = parser.parse_args()

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='reflect')

    increase_sample(args.exp+'/train/', args.train, datagen)
    increase_sample(args.exp+'/validation/', args.valid, datagen)



if __name__ == "__main__":
  main()
