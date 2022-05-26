#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=preempt \
--mail-type=END --mail-user=grigorii.sterin@tufts.edu \
--time=1-10:00:00 \
"


INPUT_FOLDER=gsterin-scratch4
GO_FOLDER=go_dir
OP_BASE=gsterin-scratch4
# DEFAULTS
SOURCE=bakers_yeast_biogrid
DEST=fission_yeast_biogrid

###### TAX_ID:
## human: 9606
## mouse: 10090
## fission: 4896
## bakers: 559292

SOURCE_ID=559292
DEST_ID=4896
NEIGHBORS=(5 10 15 20 25 30 35)
ALPHA=(0 0.05 0.1 0.2 0.4 0.6 0.8 1)
GO=(P F C)
while getopts "s:d:S:D:n:a:g:m:T" args; do
    case $args in
	s) SOURCE=${OPTARG}
	   ;;
	d) DEST=${OPTARG}
	   ;;
	S) SOURCE_ID=${OPTARG}
	   ;;
	D) DEST_ID=${OPTARG}
	   ;;
	n) NEIGHBORS=(${OPTARG})
	   ;;
	a) ALPHA=(${OPTARG})
	   ;;
	g) GO=(${OPTARG})
	   ;;
	m) MUNK=$OPTARG
	   ;;
	T) TEST=1
    esac
done

echo "UNIMUNDO Parameters: "
echo "    SOURCE=${SOURCE}, DEST=${DEST}"

OUTPUT_FOLDER=${OP_BASE}/${SOURCE}-${DEST}
OUTPUT_LOGS=${OP_BASE}/logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi

if [ -z $MUNK ]; then echo "MUNK embedding not specified; exiting..."; exit 1; fi

for G in ${GO[@]}
do
    for A in ${ALPHA[@]}
    do
	for N in ${NEIGHBORS[@]}
	do
	    OUTPUT_LOG_FILE=${OUTPUT_LOGS}/${SOURCE}-${DEST}-GO-${G}-ALPHA-${A}-NEIGHBORS-${N}.log
	    sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./src/unimundo_classify.py --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --network_source=${SOURCE} --network_target=${DEST} --munk_name=${MUNK} --go_type=${G} --src_org_id=${SOURCE_ID} --tar_org_id=${DEST_ID} --n_neighbors=${N} --verbose --alpha=${A}
	    if [ ! -z $TEST ]; then echo "Testing complete..."; exit 0; fi
	done
    done
done
    
