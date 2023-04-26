#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/multitask_vnn"

dataset="av"

folds=5

ont="ctg"

ctrl_count=83
for ((d=0;d<=ctrl_count;d++));
do
	for drug in "RS_${d}"
	do
		bash "${homedir}/scripts/create_input.sh" $homedir $ont $dataset $drug $folds
		
		for ((i=1;i<=folds;i++));
		do
			sbatch -J "MTL_VNN_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/cv_slurm.sh $homedir $ont $dataset $drug $i
			sbatch -J "MTL_VNN_${ont}_${drug}_${i}" -o "${homedir}/logs/rlipp_${ont}_${drug}_${i}.log" ${homedir}/scripts/rlipp_slurm.sh $homedir $ont $dataset $drug $i
		done
	done
done