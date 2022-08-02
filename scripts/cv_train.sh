#!/bin/bash
homedir=$1

gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"
ontfile="${homedir}/data/training_files_${3}/ontology_${2}_${3}.txt"
mutationfile="${homedir}/data/training_files_${3}/cell2mutation_${2}_${3}.txt"
cn_deletionfile="${homedir}/data/training_files_${3}/cell2cndeletion_${2}_${3}.txt"
cn_amplificationfile="${homedir}/data/training_files_${3}/cell2cnamplification_${2}_${3}.txt"
taskfile="${homedir}/data/training_files_${3}/task_list_${4}.txt"

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}"
traindatafile="${modeldir}/train.txt"
hyperparamsfile="${modeldir}/tuned_hyperparams.txt"

cudaid=0

pyScript="${homedir}/src/train_vnn.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile -train $traindatafile \
	-mutations $mutationfile -cn_deletions $cn_deletionfile -cn_amplifications $cn_amplificationfile \
	-tasks $taskfile -model $modeldir -genotype_hiddens 4 -lr 0.0005 -cuda $cudaid -epoch 300 \
	-batchsize 64 -optimize 0 -tuned_hyperparams $hyperparamsfile > "${modeldir}/train.log"

#qcscript="${homedir}/src/qc_plots.py"

#source activate base

#python -u $qcscript $modeldir