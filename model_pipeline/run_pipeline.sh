#!/usr/bin/env bash
#Fabio Zanarello, Sanger Institute, 2020

img_folder=$1
data_info=$2
exp_name=$3

################################################################################

echo
echo --------------------------------------------------------------------------
echo Creating folder for $exp_name esperiment...

mkdir $exp_name
mkdir $exp_name/train
mkdir $exp_name/validation
mkdir $exp_name/results

echo DONE!
echo

################################################################################
echo --------------------------------------------------------------------------
echo Organising files in folder...

python3 prepare_data.py --img $img_folder --info $data_info --split 0.8 --exp $exp_name

echo DONE!
echo

###############################################################################
echo --------------------------------------------------------------------------
echo Increasing sample size...

python3 increase_size.py --exp $exp_name

echo DONE!
echo

################################################################################
echo --------------------------------------------------------------------------
echo Computig the model...

python3 compute_model.py --exp $exp_name 

echo DONE!
echo

################################################################################
echo --------------------------------------------------------------------------
echo Cleaning $exp_name intermediate files...

rm -rf $exp_name/train
rm -rf $exp_name/validation

echo DONE!
echo
