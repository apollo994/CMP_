#!/usr/bin/env bash
#Fabio Zanarello, Sanger Institute, 2020

sample_folder=$1
images=$2
experiment=$3
agu_folder=$4


for tab in $sample_folder/* ; do

    name=$(basename $tab)

    python3 get_positive_label.py --info $tab --name $name

    pos_lab=`cat 'tmp_positive_label_'$name'.txt'`

    echo $pos_lab


    bsub \
          -R "select[ngpus>0 && mem>4000] rusage[ngpus_physical=1.00,mem=4000] span[gtile=1]" \
          -M4000 \
          -gpu "mode=exclusive_process" \
          -q gpu-normal \
          -J $name \
          -o $name'.out' \
          -e $name'.err' \
          "sh run_integrated_model.sh $images $tab $name'_'$experiment $pos_lab $agu_folder"

    rm 'tmp_positive_label_'$name'.txt'

done
