#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=preempt \
--mail-type=END --mail-user=grigorii.sterin@tufts.edu \
--time=1-10:00:00 \
"

DIMS=(5 10 15 20 25 30)
#DIMS=(20 50 100 150 200 250 300)
#DIMS=(20 30 50 70 100)
while getopts "s:d:m:p:" args; do
    case $args in
	s) SOURCE=$OPTARG
	   ;;
	d) DEST=$OPTARG
	   ;;
	m) MAPPING=$OPTARG
	   ;;
	p) DIMS=($OPTARG)
	   ;;
    esac
done

WORKING_FOLDER=gsterin-scratch4

for DIM in ${DIMS[@]}; do
    OUTPUT=${WORKING_FOLDER}/network_logs/MUNDO_${SOURCE}_${DEST}_${DIM}_UMUNDO.log
    sbatch $SBATCH_OPTS -o ${OUTPUT} ./src/unimundo_embed.py --working_folder=${WORKING_FOLDER} --biogrid_tsv_folder=${WORKING_FOLDER} --source_organism_name=${SOURCE} --target_organism_name=${DEST} --mapping=${MAPPING} --mapping_num_of_pairs=${DIM} --construct_dsd --construct_dsd_dist --construct_kernel --save_munk_matrix --verbose
done
