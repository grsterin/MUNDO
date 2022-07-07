#!/bin/bash

SBATCH_OPTS="--mem=64Gb \
--partition=preempt \
--time=1-00:00:00"

echo $(pwd)

OUTPUT=sample.output
sbatch $SBATCH_OPTS -o $OUTPUT r_isorank_sample.sh  final-sample-output.txt isorank_sample_data/data.inp

