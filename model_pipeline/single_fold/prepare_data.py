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

    args = parser.parse_args()

    df = pd.read_csv(args.info)

    cls = list(set(df.ft))

    #two classes
    cl0 = cls[0]
    cl1 = cls[1]

    cl0_img = list(df[df.ft == cl0].im_id)
    cl1_img = list(df[df.ft == cl1].im_id)

    #sizes of the training and validation relative to the two classes
    cl0_train_size = round(len(cl0_img) * float(args.split))
    cl1_train_size = round(len(cl1_img) * float(args.split))

    train_im_cl0 = random.sample(cl0_img, cl0_train_size)
    validation_im_cl0 = [im for im in cl0_img if im not in train_im_cl0]

    train_im_cl1 = random.sample(cl1_img, cl1_train_size)
    validation_im_cl1 = [im for im in cl1_img if im not in train_im_cl1]

    print ()
    print ('Training size = ', len(train_im_cl0)+len(train_im_cl1))
    print (cl0, '=', len(train_im_cl0))
    print (cl1, '=', len(train_im_cl1))
    print ()
    print ('Validation size = ', len(validation_im_cl0)+len(validation_im_cl1))
    print (cl0, '=', len(validation_im_cl0))
    print (cl1, '=', len(validation_im_cl1))
    print ()

    #Creating subfolders
    os.mkdir(args.exp+'/train/'+cl0)
    os.mkdir(args.exp+'/train/'+cl1)
    os.mkdir(args.exp+'/validation/'+cl0)
    os.mkdir(args.exp+'/validation/'+cl1)

    #Linking images
    link_images(args.img, train_im_cl0, cl0, args.exp+'/train/'+cl0)
    link_images(args.img, train_im_cl1, cl1, args.exp+'/train/'+cl1)
    link_images(args.img, validation_im_cl0, cl0, args.exp+'/validation/'+cl0)
    link_images(args.img, validation_im_cl1, cl1, args.exp+'/validation/'+cl1)

    print("Images linked to feature folders")


if __name__ == "__main__":
  main()
