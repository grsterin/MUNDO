#!/bin/bash

SBATCH_OPTS="\
--mem=64GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"


while getopts "s:t:q:S:T:o:r" args; do
    case $args in
	s) SOURCE=$OPTARG
	   ;;
	t) TARGET=$OPTARG
	   ;;
	q) SEQUENCE=$OPTARG
	   ;;
	S) SOURCEI=$OPTARG
	   ;;
	T) TARGETI=$OPTARG
	   ;;
	o) OUTPUT=data/network/isorank/$OPTARG
	   ;;
	r) TEST=1
	;;
    esac
done

OUTPUT_LOG=logs/${OUTPUT##*/}
echo "OUTPUT: ${OUTPUT_LOG}"

if [ ! -d logs ]; then mkdir logs; fi

if [ -z $TEST ]; then
    sbatch $SBATCH_OPTS -o ${OUTPUT_LOG} ./isorank_compute.py --source $SOURCE --target $TARGET --sequence_score $SEQUENCE --sequence_score_src $SOURCEI --sequence_score_tar $TARGETI --output_mapping=$OUTPUT --verbose
else
     ./isorank_compute.py --source $SOURCE --target $TARGET --sequence_score $SEQUENCE --sequence_score_src $SOURCEI --sequence_score_tar $TARGETI --output_mapping=$OUTPUT --verbose
fi


#Test run
# ./run_isorank.sh -s data/networks/human_12000_biogrid.tsv -t data/networks/mouse_12000_biogrid.tsv -S human -T mouse -q seq_data/human-mouse/human-mouse-seq-sim-entrezgene.tsv -o human-mouse_it_3_al_0.5.tsv -t
