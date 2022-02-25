#!/bin/bash

SBATCH_OPTS="\
--mem=64GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"

DIM=1000
while getopts "n:d:" args; do
    case $args in
	n) NETWORK=$OPTARG
	   ;;
	d) DIM=$OPTARG
	   ;;
    esac
done

WORKING_FOLDER=/cluster/tufts/cowenlab/Projects/UNIMUNDO/MUNDO/training_dir
OUTPUT=$WORKING_FOLDER/network_logs/${NETWORK}_${DIM}.log

sbatch $SBATCH_OPTS -o ${OUTPUT} ./mashup_embedding.py --working_folder=${WORKING_FOLDER} --network=${NETWORK} --dim=${DIM}
