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
MASHUP_EMB=
ORG_ID=

while getopts "m:o:g:" args; do
    case $args in
	m) MASHUP_EMB=${OPTARG}
	   ;;
	o) ORG_ID=${OPTARG}
	   ;;
	g) GO_TYPE=${OPTARG}
	   ;;
    esac
done

OUTPUT_FOLDER=${OP_BASE}/${MASHUP_EMB}
OUTPUT_LOGS=${OP_BASE}/logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi

if [ -z $MASHUP_EMB ] || [ -z $ORG_ID ]; then echo "Either Organism I.D. or MASHUP embedding not provided. exiting..."; exit 1; fi


for G in ${GO_TYPE[@]}
do
    OUTPUT_LOG_FILE=${OUTPUT_LOGS}/GO-${G}.log
    sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./src/mashup_classify.py --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --network_source=${SOURCE} --network_target=${DEST} --munk_name=${MUNK} --go_type=${G} --src_org_id=${SOURCE_ID} --tar_org_id=${DEST_ID} --n_neighbors=${N} --verbose --alpha=${A}
done
