#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import os
import glob
import argparse
from collections import defaultdict

import random
import pandas as pd
import numpy as np

def link_agumented(im_list, fold, exp, pos, agu):
    for i in im_list:
        i = i.strip('.jpg')

        i_out = i + '_outlines.jpg'
        i_over = i + '_overlay.jpg'
        i_flo = i + '_flowi.jpg'

        images_to_link = [i_out, i_over, i_flo]

        for im in images_to_link:
            from_path = os.path.abspath(f'{agu}/{im}')
            to_path = f'{exp}/{fold}/train/{pos}/{im}'
            try:
                os.symlink(from_path, to_path)
            except OSError:
                print (f'Linking failed with {im}')
            else:
                pass

def print_status(exp, fold, pos):

    pos_len=len(glob.glob(f'{exp}/{fold}/train/{pos}/*.jpg'))
    neg_len=len(glob.glob(f'{exp}/{fold}/train/Not_{pos}/*.jpg'))
    print (f'Examples in {fold}:')
    print (f'{pos} = {pos_len}')
    print (f'Not_{pos} = {neg_len}')


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')
    parser.add_argument('--pos', metavar='POS', help='positive label')
    parser.add_argument('--agu', metavar='AGU', help='agumented images folder')

    args = parser.parse_args()

    folds = sorted([x.split('/')[-1] for x in glob.glob(f'{args.exp}/*') if 'results' not in x])


    for fold in folds:

        print_status(args.exp, fold, args.pos)
        print('Linking segmented images...')
        print()
        pos_list = [x.split('/')[-1] for x in glob.glob(f'{args.exp}/{fold}/train/{args.pos}/*.jpg')]
        link_agumented(pos_list, fold, args.exp, args.pos, args.agu)

        #neg_list = [x.split('/')[-1] for x in glob.glob(f'{args.exp}/{fold}/train/Not_{args.pos}/*.jpg')]
        #link_agumented(neg_list, fold, args.exp, f'Not_{args.pos}', args.agu)

        print_status(args.exp, fold, args.pos)

        print (f'####################################################')


if __name__ == "__main__":
  main()
