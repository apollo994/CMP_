#!/usr/bin/env python3

# Fabio Zanarello
# Cambridge, 2020

#This file contains all the custom functions designed for the CMP image classifier project

import os

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

def create_folders(class_list, folder):
    for cl in class_list:
        try:
            os.mkdir(folder+'/'+cl)
        except OSError:
            print ("Creation of the directory %s failed" % folder+'/'+cl)
        else:
            print ("Successfully created the directory %s " % folder+'/'+cl)

################################################################################
#image link in calss folder depending on img name

def link_images(img_path, name_to_class, target_folder):
    for img in name_to_class.keys():
        cl = name_to_class[img]

        from_path = os.path.abspath(img_path+"/"+img+".jpg")
        to_path = target_folder+"/"+cl+"/"+img+".jpg"


        os.symlink(from_path, to_path)
