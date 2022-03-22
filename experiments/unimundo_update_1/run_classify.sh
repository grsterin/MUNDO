#!/bin/bash

SBATCH_OPTS="\
--mem=32GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"


INPUT_FOLDER=network_dir
GO_FOLDER=go_dir
OP_BASE=output_dir
# DEFAULTS
SOURCE=human_12000_biogrid
DEST=mouse_12000_biogrid

###### TAX_ID:
## human: 9606
## mouse: 10090
## fission: 4896
## bakers: 559292

SOURCE_ID=9606
DEST_ID=10090
NEIGHBORS=(10 20 30 50)
M_NEIGHBORS=15 # 10 15 20 25 30 35)
ALPHA=(0 0.2 0.4 0.6 0.8 1)
GO=(P F C)
MAPPING=(hubalign)
DIM=(20 50 100 150 200 250 300)

while getopts "s:d:S:D:n:a:g:M:Tbh:" args; do
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
	T) TEST=1
	   ;;
	M) M_NEIGHBORS=(${OPTARG})
	   ;;
	b) MAPPING+=(hubalign_blast)
	   ;;
	h) DIM=(${OPTARG})
	   ;;
    esac
done

echo "UNIMUNDO Parameters: "
echo "    SOURCE=${SOURCE}, DEST=${DEST}"

OUTPUT_FLD=${OP_BASE}/${SOURCE}-${DEST}
OUTPUT_LOGS=${OP_BASE}/logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi

# if [ -z $MUNK ]; then echo "MUNK embedding not specified; exiting..."; exit 1; fi

for M in ${MAPPING[@]}
do
    OUTPUT_FOLDER=${OUTPUT_FLD}/${M}
    if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi
    echo "Mapping:= ${M}"
    for G in ${GO[@]}
    do
	for A in ${ALPHA[@]}
	do
	    for N in ${NEIGHBORS[@]}
	    do
		for D in ${DIM[@]}
		do
		    OUTPUT_LOG_FILE=${OUTPUT_LOGS}/${SOURCE}-${DEST}-${M}-GO-${G}-ALPHA-${A}-NEIGHBORS-${N}-MUNK-DIMS-${D}.log
		    sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./src/unimundo_classify.py --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --network_source=${SOURCE} --network_target=${DEST} --munk_name=${M} --go_type=${G} --src_org_id=${SOURCE_ID} --tar_org_id=${DEST_ID} --n_neighbors=${N} --verbose --alpha=${A} --n_neighbors_munk=${M_NEIGHBORS} --munk_dim=${D}
		    if [ ! -z $TEST ]; then echo "Testing complete..."; exit 0; fi
		done
	    done
	done
    done
done
