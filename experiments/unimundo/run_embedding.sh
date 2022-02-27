#!/bin/bash

SBATCH_OPTS="\
--mem=64GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"

DIM=300
while getopts "s:d:m:p:" args; do
    case $args in
	s) SOURCE=$OPTARG
	   ;;
	d) DEST=$OPTARG
	   ;;
	m) MAPPING=$OPTARG
	   ;;
	p) DIM=$OPTARG
	   ;;
    esac
done

WORKING_FOLDER=/cluster/tufts/cowenlab/Projects/UNIMUNDO/MUNDO/training_dir
OUTPUT=$WORKING_FOLDER/network_logs/${NETWORK}_${DIM}_UMUNDO.log

sbatch $SBATCH_OPTS -o ${OUTPUT} ./unimundo_embed.py --working_folder=${WORKING_FOLDER} --biogrid_tsv_folder=${WORKING_FOLDER} --source_organism_name=${SOURCE} --target_organism_name=${DEST} --mapping=${MAPPING} --mapping_num_of_pairs=${DIM} --construct_dsd --construct_dsd_dist --construct_kernel --save_munk_matrix --verbose
