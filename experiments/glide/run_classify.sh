#!/bin/bash

SBATCH_OPTS="\
--mem=32GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"


INPUT_FOLDER=net
GO_FOLDER=go
OP_BASE=output_folder
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
S_NEIGHBORS=(10 20 30 50)
T_NEIGHBORS=(10 15 20 25 30 35)
ALPHA=(0 0.2 0.4 0.6 0.8 1)
GO=(P F C)
MAPPING=hubalign+blast
DIM=(20 50 100 150 200 250 300 500 1000 2000)

while getopts "s:d:S:D:n:a:g:M:Tb:h:" args; do
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
	b) MAPPING=${OPTARG}
	   ;;
	h) DIM=(${OPTARG})
	   ;;
    esac
done

echo "UNIMUNDO Parameters: "
echo "    SOURCE=${SOURCE}, DEST=${DEST}"
LANDMARK_FILE=${DEST}-${SOURCE}-${MAPPING}.tsv

OUTPUT_FLD=${OP_BASE}/${SOURCE}-${DEST}
OUTPUT_LOGS=${OP_BASE}/logs

if [ ! -d ${OUTPUT_LOGS} ]; then mkdir ${OUTPUT_LOGS}; fi
if [ ! -d ${OUTPUT_FLD} ]; then mkdir ${OUTPUT_FLD}; fi

# if [ -z $MUNK ]; then echo "MUNK embedding not specified; exiting..."; exit 1; fi
# python glide_create_embedding.py --input_folder net --go_folder go --output_folder . --network_source bakers_yeast_biogrid --network_target fission_yeast_biogrid --landmark_file fission-yeast-bakers-yeast-with-blast.alignment.tsv --landmark_no 100 --src_org_id 559292 --tar_org_id 4896

OUTPUT_FOLDER=${OUTPUT_FLD}/${MAPPING}
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir ${OUTPUT_FOLDER}; fi
echo "Mapping:= ${MAPPING}"
for G in ${GO[@]}
do
	for A in ${ALPHA[@]}
	do
	    for M in ${S_NEIGHBORS[@]}
	    do
			for N in ${T_NEIGHBORS[@]}
			do
				for D in ${DIM[@]}
				do
		    		OUTPUT_LOG_FILE=${OUTPUT_LOGS}/${SOURCE}-${DEST}-${M}-GO-${G}-ALPHA-${A}-NEIGHBORS-${N}-MUNK-DIMS-${D}.log
		    		sbatch $SBATCH_OPTS -o ${OUTPUT_LOG_FILE} ./glide_create_embedding.py --landmark_file=${LANDMARK_FILE} --input_folder=${INPUT_FOLDER} --go_folder=${GO_FOLDER} --output_folder=${OUTPUT_FOLDER} --network_source=${SOURCE} --network_target=${DEST} --landmark_file=${MAPPING} --go_type=${G} --src_org_id=${SOURCE_ID} --tar_org_id=${DEST_ID} --target_neighbors=${N} --verbose --alpha=${A} --source_neighbors=${M} --landmark_no ${D}
		    		if [ ! -z $TEST ]; then echo "Testing complete..."; exit 0; fi
				done
			done
	    done
    done
done
