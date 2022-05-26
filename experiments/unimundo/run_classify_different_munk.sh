#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"


INPUT_FOLDER=gsterin-scratch
GO_FOLDER=go_dir
OP_BASE=output_dir
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

NEIGHBORS=(5 15 25 35)
ALPHA=(0 0.1 0.4 0.8 1)
GO=(P F C)

#different MUNK variations

MUNKDIM=(20 30 50 70 100)
MUNKPREF="fission-yeast-bakers-yeast-pure-blast-landmarks.dim"
MUNKS=(`for d in ${MUNKDIM[@]}; do echo "${MUNKPREF}_${d}.lap_0.1.munk"; done`)
OUTPUT_PREFIX=blast:

while getopts "s:d:S:D:n:a:g:m:T:p:" args; do
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
	m) MUNKS=($OPTARG)
	   ;;
	T) TEST=1
	   ;;
	p) OUTPUT_PREFIX=${OPTARG}
	   ;;
    esac
done

echo "UNIMUNDO Parameters: "
echo "    SOURCE=${SOURCE}, DEST=${DEST}"

OUTPUT_FOLDER=${OP_BASE}/${OUTPUT_PREFIX}${SOURCE}-${DEST}
OUTPUT_LOGS=${OP_BASE}/${OUTPUT_PREFIX}logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir -p ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p ${OUTPUT_FOLDER}; fi


if [ ! -f ${INPUT_FOLDER}/${SOURCE}.dsd.rbf_0.1.npy ]; then echo "Source DSD absent"; exit 1; fi

if [ ! -f ${INPUT_FOLDER}/${DEST}.dsd.rbf_0.1.npy ]; then echo "Target DSD absent"; exit 1; fi

for MUNK in ${MUNKS[@]}; do if [ ! -f ${INPUT_FOLDER}/${MUNK}.npy ]; then echo "MUNK embedding not specified; exiting..."; exit 1; fi; done;
echo "All MUNK embeddings present. Continuing..."

for G in ${GO[@]}
do
    for A in ${ALPHA[@]}
    do
	for N in ${NEIGHBORS[@]}
	do
	    for MUNK in ${MUNKS[@]}
	    do
		LANDMARK=${MUNK#*_}
		LANDMARK=${LANDMARK%%.*}
		OUTPUT_LOG_FILE=${OUTPUT_LOGS}/${SOURCE}-${DEST}-GO-${G}-ALPHA-${A}-NEIGHBORS-${N}_LANDMARK_${LANDMARK}.log	        
		sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./src/unimundo_classify.py --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --network_source=${SOURCE} --network_target=${DEST} --munk_name=${MUNK} --go_type=${G} --src_org_id=${SOURCE_ID} --tar_org_id=${DEST_ID} --n_neighbors=${N} --verbose --alpha=${A} --landmark_no=${LANDMARK}
		# if [ ! -z $TEST ]; then echo "Testing complete..."; exit 0; fi
	    done
	done
    done
done
    
