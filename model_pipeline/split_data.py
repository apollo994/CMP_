#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import os
import argparse
from collections import defaultdict

import random
import pandas as pd
import numpy as np

def link_images(img_path, list_img, cl, target):
    for img in list_img:
        from_path = os.path.abspath(img_path+"/"+img+".jpg")
        to_path = target+"/"+img+".jpg"
        try:
            os.symlink(from_path, to_path)
        except OSError:
            print (f'Link already exists')
        else:
            pass


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--img', metavar='IMG_FOLDER', help='path to the img folder')
    parser.add_argument('--info', metavar='INFO_TAB', help='table containig img name and feature')
    parser.add_argument('--split', metavar='SPLIT', default = 0.8 ,  help='proportion of training vs validation')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')
    parser.add_argument('--k', metavar='K', default = 5 ,  help='number of fold')

    args = parser.parse_args()

    df = pd.read_csv(args.info)

    cls = list(set(df.ft))

    #two classes
    cl0 = cls[0]
    cl1 = cls[1]

    cl0_img = list(df[df.ft == cl0].im_id)
    cl1_img = list(df[df.ft == cl1].im_id)

    cl0_lists = [list(li) for li in np.array_split(np.array(cl0_img),args.k)]
    cl1_lists = [list(li) for li in np.array_split(np.array(cl1_img),args.k)]

    print (f'Found {len(cl0_img)+len(cl1_img)} imges')
    print (f'{cl0}: {len(cl0_img)}')
    print (f'{cl1}: {len(cl1_img)}')
    print ()

    fold = 0

    for c0, c1 in zip(cl0_lists, cl1_lists):

        train_im_cl0 = [im for im in cl0_img if im not in c0]
        train_im_cl1 = [im for im in cl1_img if im not in c1]
        validation_im_cl0 = c0
        validation_im_cl1 = c1
        fold += 1

        print (f'Creating fold {fold}...')
        print ('Training set:')
        print (f'{cl0}: {len(train_im_cl0)}')
        print (f'{cl1}: {len(train_im_cl1)}')
        print ('Validation set:')
        print (f'{cl0}: {len(validation_im_cl0)}')
        print (f'{cl1}: {len(validation_im_cl1)}')
        print ()

        os.mkdir(args.exp+f'/fold_{fold}/')
        os.mkdir(args.exp+f'/fold_{fold}/train/')
        os.mkdir(args.exp+f'/fold_{fold}/validation/')
        os.mkdir(args.exp+f'/fold_{fold}/'+'/train/'+cl0)
        os.mkdir(args.exp+f'/fold_{fold}/'+'/train/'+cl1)
        os.mkdir(args.exp+f'/fold_{fold}/'+'/validation/'+cl0)
        os.mkdir(args.exp+f'/fold_{fold}/'+'/validation/'+cl1)

        link_images(args.img, train_im_cl0, cl0, args.exp+f'/fold_{fold}/'+'/train/'+cl0)
        link_images(args.img, train_im_cl1, cl1, args.exp+f'/fold_{fold}/'+'/train/'+cl1)
        link_images(args.img, validation_im_cl0, cl0, args.exp+f'/fold_{fold}/'+'/validation/'+cl0)
        link_images(args.img, validation_im_cl1, cl1, args.exp+f'/fold_{fold}/'+'/validation/'+cl1)


if __name__ == "__main__":
  main()
