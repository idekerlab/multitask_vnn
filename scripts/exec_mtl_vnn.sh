#!/bin/bash

homedir="" # homedir

dataset="av"

folds=5

ont="ctg"

ct_data=""

for drug in "RS"
do
	# ALWAYS keep it commented unless you want to train new models
	# bash "${homedir}/scripts/create_input.sh" $homedir $ont $dataset $drug $folds
		
	for ((i=1;i<=folds;i++));
	do
		#sbatch -J "MTL_VNN_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_slurm.sh "$homedir" "$ont" "$dataset" "$drug" "$i" "$ct_data"
		sbatch -J "MTL_VNN_${ont}_${drug}_${i}" -o "${homedir}/logs/rlipp_${ont}_${drug}_${i}.log" ${homedir}/scripts/rlipp_slurm.sh "$homedir" "$ont" "$dataset" "$drug" "$i" "$ct_data"
	done
done