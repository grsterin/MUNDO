#!/bin/bash
#SBATCH --job-name=rpblast
#SBATCH --output=rpblast.output
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=preempt

BLAST_LOC=../binaries/ncbi-blast-2.13.0+/bin/blastp
SOURCE_FASTA=
TARGET_FASTA=
FORWARD_OUT=
REVERSE_OUT=
# final tsv file for reciprocal blast
FINAL_TSV=

USAGE () {
    echo 'USAGE:'
    echo '       ./run_rpblast.sh -s [SOURCE-FASTA-FILE] -t [TARGET-FASTA-FILE] -f [OUTPUT-FILE-LOCATION-FORWARD] -r [OUTPUT-FILE-LOCATION-REVERSE] -o [OUTPUT-TSV-FILE]'
}  


while getopts "s:t:f:r:o:" arg; do
    case ${arg} in
	s) SOURCE_FASTA=${OPTARG}
	   ;;
	t) TARGET_FASTA=${OPTARG}
	   ;;
	f) FORWARD_OUT=${OPTARG}
	   ;;
	r) REVERSE_OUT=${OPTARG}
	   ;;
	o) FINAL_TSV=${OPTARG}
    esac
done

if [ -z $SOURCE_FASTA ] || [ -z $TARGET_FASTA ] || [ -z $FORWARD_OUT ] || [ -z $REVERSE_OUT ] || [ -z $FINAL_TSV ]
then
    USAGE
    exit 1
fi

# Forward run and backward run and wait
srun -n 1 $BLAST_LOC -out $FORWARD_OUT -outfmt 6 -query $SOURCE_FASTA -subject $TARGET_FASTA -num_threads 5 &
srun -n 1 $BLAST_LOC -out $REVERSE_OUT -outfmt 6 -query $TARGET_FASTA -subject $SOURCE_FASTA -num_threads 5 &
wait

srun -n 1 python perform_reciprocal_blast.py --out_forward $FORWARD_OUT --out_reverse $REVERSE_OUT --final_op $FINAL_TSV --verbose

