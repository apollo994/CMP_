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

    args = parser.parse_args()


    info = pd.read_csv(args.info)

    all_ft = set(info.ft)


    with open ('tmp_positive_label.txt', 'w') as p_lab:
        for f in all_ft:
            if 'Not' not in f:
                p_lab.write(f)





if __name__ == "__main__":
  main()
