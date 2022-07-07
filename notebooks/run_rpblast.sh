#!/bin/bash
SBATCH_OPTS="--mem 64GB --nodes 1 --ntasks 30 --partition preempt --time 1-00:00:00"

LOGFILE="op.log.txt"
sbatch -o $LOGFILE ./perform_reciprocal_blast.py --blast_loc ../binaries/ncbi-blast-2.13.0+/bin/blastp --no_threads_for_blast 5 --source_fasta blast-p-data-in-human-yeast/biogrid_Human_ID_9606_sequences.fasta --target_fasta blast-p-data-in-human-yeast/biogrid_Yeast_ID_559292_sequences.fasta --out_forward forward.log --out_reverse reverse.log --final_op output.tsv --verbose



