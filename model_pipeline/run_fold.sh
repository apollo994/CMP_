#!/usr/bin/env bash
#Fabio Zanarello, Sanger Institute, 2020

img_folder=$1
data_info=$2
exp_name=$3
positive_label=$4

################################################################################

echo
echo --------------------------------------------------------------------------
echo Creating folder for $exp_name esperiment...

mkdir $exp_name
mkdir $exp_name/results

echo DONE!
echo

################################################################################

echo
echo --------------------------------------------------------------------------
echo Splitting data in chunks...

python3 split_data.py --img $img_folder --info $data_info --exp $exp_name

echo DONE!
echo

################################################################################

echo
echo --------------------------------------------------------------------------
echo Increasing sample size...

python3 increase_fold.py --exp $exp_name

echo DONE!
echo

###############################################################################

echo
echo --------------------------------------------------------------------------
echo Computing models...

python3 compute_fold.py --exp $exp_name --info $data_info --pos $positive_label

echo DONE!
echo

################################################################################

echo
echo --------------------------------------------------------------------------
echo Plotting...

python3 get_plot.py --exp $exp_name

echo DONE!
echo

################################################################################
echo --------------------------------------------------------------------------
echo Cleaning $exp_name intermediate files...

rm -rf $exp_name/fold*

echo DONE!
echo
