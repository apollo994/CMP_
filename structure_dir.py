#!/usr/bin/env python3

# Fabio Zanarello
# Cambridge, 2020

import sys
import os
import argparse
from collections import defaultdict
import pandas as pd
from CMP_functions import *


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--img', metavar='DIR', default="/dev/stdin", help='The directory containig all images')
    parser.add_argument('--info', metavar='INFO', help='Table with images information')
    parser.add_argument('--ft', metavar='FT', help='Features in the info table you want to split on')
    parser.add_argument('--ft_min', metavar='FT_MIN', default=1 , type=int, help='Minimum number of example per class')
    parser.add_argument('--fold', metavar='FOLD', help='Where to put the precessed folders')

    args = parser.parse_args()



    img_info = pd.read_csv(args.info)

    class_count, class_list = get_classes(img_info, args.ft, args.ft_min)

    name_to_class = get_img_dict(img_info, class_list, args.ft)

    #create_set_folder(set=['train','test','val'], size=[0.8,0.1,0.1])

    create_class_folders(class_list, args.fold)

    link_images(args.img, name_to_class, args.fold)




if __name__ == "__main__":
  main()
