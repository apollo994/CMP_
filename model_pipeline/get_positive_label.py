#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import argparse

import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--info', metavar='INFO_TAB', help='table containig img name and feature')
    parser.add_argument('--name', metavar='NAME', help='name of the sample')

    args = parser.parse_args()


    info = pd.read_csv(args.info)

    all_ft = set(info.ft)

    for f in all_ft:
        if 'Not' not in f:
            with open (f'tmp_positive_label_{args.name}.txt', 'w') as p_lab:
                p_lab.write(f)





if __name__ == "__main__":
  main()
