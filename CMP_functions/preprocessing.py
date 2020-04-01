#!/usr/bin/env python3

# Fabio Zanarello
# Cambridge, 2020

#This file contains all the custom functions designed for the CMP image procesing steps

import os
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import Image as pymage
from random import sample


##################################################################################
#List file in directory

def get_files(path):

    names = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            name = filename.split('.')[0]
            if len(name)>0:
                names.append(name)
    return names

##################################################################################
#Get header of csv from path and return DataFrame

def get_head(path):
    im_info = pd.read_csv(path)
    print (im_info.head())
    return im_info

#################################################################################
def check_duplicates(l, name):
    if len(l)>len(set(l)):
        n = len(l)-len(set(l))
        print (f'{name} has {n} duplicates!')
#     else:
#         print (f'No duplicates in {name}!')


################################################################################
#Check if image files corresponds to file nem in info table

def check_consistency(im_names, im_info):

    to_keep = []
    missing_file = []
    missing_info = []

    info_names = list(im_info['im_id'])

    check_duplicates(info_names, 'info_table')
    check_duplicates(im_names, 'images folder')
    print()

    n_images_file = len(im_names)
    n_images_info = len(info_names)

    for im in im_names:
        if im in info_names:
            to_keep.append(im)
        else:
            missing_info.append(im)

    for info in info_names:
        if info in im_names:
            to_keep.append(im)
        else:
            missing_file.append(info)

    to_keep = list(set(to_keep))

    print (f'Found {len(to_keep)} images-info matches out of {n_images_file} images and {n_images_info} info')
    print (f'{len(missing_file)} missing file found')
    print (f'{len(missing_info)} missing info found')

    return to_keep, missing_file, missing_info


##################################################################################
#Create folder and save converted jpg

def convert_tif(tif_folder, data_folder):
    c = 0

    PATH_jpg = data_folder+'img_jpg/'

    try:
        os.mkdir(PATH_jpg)
        print("Directory " , PATH_jpg ,  " Created ")
    except FileExistsError:
        print("Directory " , PATH_jpg ,  " already exists")


    for name in glob.glob(tif_folder+'/*.tif'):
        im = Image.open(name)
        name = str(name).rstrip(".tif").split('/')[-1]

        im.save(PATH_jpg+name + '.jpg', 'JPEG')

        c+=1

    print (f"Converted {c} images and saved at {PATH_jpg} ")

    return (PATH_jpg)


################################################################################
#Get list of calss in feature and count dictionary of observations per class

def get_classes(img_info, feature, ft_min):

    all_class_count = dict(img_info[feature].value_counts())

    class_count = {}
    class_list = []

    for ft in all_class_count.keys():
        if all_class_count[ft]<ft_min:
            print (ft, "not considered")
        else:
            class_count[ft]=all_class_count[ft]
            class_list.append(ft)

    del all_class_count

    return class_count, class_list

################################################################################
#Get dictionary from image id to class

def get_img_dict(img_info, class_list,feature):

    cleaned_img_info = img_info[img_info[feature].isin(class_list)]

    name_to_class = dict(zip(cleaned_img_info['im_id'], cleaned_img_info[feature]))

    return name_to_class

################################################################################
#create folder with names contained in list

def create_class_folders(class_list, feature, data_dir):
    try:
        os.mkdir(data_dir+feature)
    except OSError:
        print (f'Directory {feature} already exists')
    else:
        print (f'Successfully created the directory {feature}')

    for cl in class_list:
        try:
            os.mkdir(data_dir+feature+'/'+cl)
        except OSError:
            print (f'Sub-directory {cl} already exists')
        else:
            print (f'Successfully created the sub-directory {cl}')

################################################################################
#image link in calss folder depending on img name

def link_images(img_path, name_to_class, target_folder):
    for img in name_to_class.keys():
        cl = name_to_class[img]

        from_path = os.path.abspath(img_path+"/"+img+".jpg")
        to_path = target_folder+"/"+cl+"/"+img+".jpg"

        try:
            os.symlink(from_path, to_path)
        except OSError:
            print (f'Link already exists')
        else:
            print (f'Link successfully created')
    print("Images linked to feature folders")

################################################################################
#get overview of a features giving the class count dictionary

def get_overview(feature, class_count):

    #print (f'Number of images: {sum(class_count.values())}')
    print (f'{sum(class_count.values())} images are subdividen in {len(class_count)} {feature}')
    print ()
    for f in class_count:
        print (f'{f}: {class_count[f]}')

    feature_df = pd.DataFrame.from_dict(class_count,orient='index')
    feature_df.rename(columns = {0:'img_count'}, inplace = True)

    fig_dims = (12, 8)
    fig, ax = plt.subplots(figsize=fig_dims)

    chart = sns.barplot(x = feature_df.index,
                        y = "img_count",
                        ax=ax,
                        data=feature_df,
                        palette="deep")

    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title(f'Number of images per {feature}')

    return fig

################################################################################
#plot feature distribution plot

def save_fig(fig, path, feature):
    fig.savefig(path+f'{feature}_distribution.png')

################################################################################
#Print classes

def print_classes(class_list):
    print ('Available classes:\n')
    for cl in class_list:
        print (cl)

################################################################################
#print images and info givinf features and class
#n is the number of images printed (randomly choosen)
#if want all n='a'


def view_image(cl, feature, im_info, n, path_jpg):
    list_of_images = list(im_info[im_info[feature].isin([cl])]['im_id'])

    if n=='a':
        n = len(list_of_images)
    if n>len(list_of_images):
        n = len(list_of_images)
        print ('asked more images than exist, set n=\'a\'(all)')

    selected = sample(list_of_images, n)

    for im in selected:
        file = path_jpg+im+'.jpg'
        pil_img = pymage(filename=file)
        img_name = im.split('/')[-1]
        img_name = img_name.split('.')[0]
        info = im_info[im_info['im_id']==img_name]

        tissue = info.tissue.values[0]
        cancer_type = info.cancer_type.values[0]
        model_name = info.model_name.values[0]
        SIDM = info.SIDM.values[0]


        print('YOU ARE LOOKIG AT:')
        print(f'Image name: {img_name}\nTissue: {tissue}\nCancer type:{cancer_type}\nModel name: {model_name}')
        print('\nFOR MORE INFO CHCK:')
        print(f'https://cellmodelpassports.sanger.ac.uk/passports/{SIDM}')

        display(pil_img)
