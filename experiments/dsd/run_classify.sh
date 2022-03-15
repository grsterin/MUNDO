#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"


INPUT_FOLDER=network_dir
GO_FOLDER=go_dir
OP_BASE=output_dir

GO_TYPE=(P F C)
RBF=
ORG_ID=
NEIGHBORS=(5 10 15 20 25 30 35 50)

while getopts "m:o:g:" args; do
    case $args in
	m) RBF=${OPTARG}
	   ;;
	o) ORG_ID=${OPTARG}
	   ;;
	g) GO_TYPE=${OPTARG}
	   ;;
    esac
done

OUTPUT_FOLDER=${OP_BASE}/${RBF}
OUTPUT_LOGS=${OP_BASE}/logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi

if [ -z $RBF ] || [ -z $ORG_ID ]; then echo "Either Organism I.D. or DSD RBF not provided. exiting..."; exit 1; fi


for G in ${GO_TYPE[@]}
do
    for N in ${NEIGHBORS[@]}
    do
	OUTPUT_LOG_FILE=${OUTPUT_LOGS}/GO-${G}_NEIGHBORS-${N}.log
	sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./src/dsd_classify.py --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --organism_name=${RBF}  --go_type=${G} --org_id=${ORG_ID}  --verbose --n_neighbors=${N} 
    done
done
