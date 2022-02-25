#!/bin/bash

SBATCH_OPTS="\
--mem=64GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"

DIM=100
while getopts "n:d:" args; do
    case $args in
	n) NETWORK=$OPTARG
	   ;;
	d) DIM=$OPTARG
	   ;;
    esac
done

WORKING_FOLDER=/cluster/tufts/cowenlab/Projects/UNIMUNDO/MUNDO/training_dir
OUTPUT=$WORKING_FOLDER/network_logs/${NETWORK}_${DIM}_N2VEC.log

sbatch $SBATCH_OPTS -o ${OUTPUT} ./n2vec_embed.py --working_folder=${WORKING_FOLDER} --network=${NETWORK} --dim=${DIM}
