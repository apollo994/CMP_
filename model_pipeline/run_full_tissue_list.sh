#!/usr/bin/env bash
#Fabio Zanarello, Sanger Institute, 2020

sample_folder=$1
images=$2


for tab in $sample_folder/* ; do

    name=$(basename $tab)

    python3 get_positive_label.py --info $tab

    pos_lab=`cat tmp_positive_label.txt`

    echo $pos_lab

    sh run_fold.sh $images $tab $name'_results' $pos_lab

    # bsub \
    #       -R "select[ngpus>0 && mem>1000] rusage[ngpus_physical=1.00,mem=1000] span[gtile=1]" \
    #       -M1000 \
    #       -gpu "mode=exclusive_process" \
    #       -q gpu-normal \
    #       -J $name \
    #       -o $name'.out' \
    #       -e $name'.err' \
    #       "sh run_fold.sh $images $tab $name'_results' $pos_lab"

done
