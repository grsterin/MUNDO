#!/bin/bash

./unimundo_classify.py --input_folder=../../training_dir --go_folder=../../../datasets/go_files --output_folder=./bakers-fission-results --network_source=bakers_yeast_biogrid --network_target=fission_yeast_biogrid --munk_name=fission-yeast-bakers-yeast-with-blast.alignment_lap_ker_0.1.munk --src_org_id=559292 --tar_org_id=4896 --verbose 
