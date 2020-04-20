#!/usr/bin/env bash
#Fabio Zanarello, Sanger Institute, 2020

img_folder=$1
data_info=$2
exp_name=$3

################################################################################


for i in {1..10}
do
   echo bsub -q gpu-normal \
        -J ${exp_name}_${i} \
        -o out/${exp_name}_${i}.out \
        -e err/${exp_name}_${i}.err \
        -gpu \
        - sh run_pipeline.sh $img_folder $data_info ${exp_name}_${i}
done
