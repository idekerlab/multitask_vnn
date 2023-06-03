#!/bin/bash

homedir="your_home_directory"

gene2idfile="${homedir}/sample/gene2ind.txt"
cell2idfile="${homedir}/sample/cell2ind.txt"
ontfile="${homedir}/sample/ontology.txt"
mutationfile="${homedir}/sample/cell2mutation.txt"
cn_deletionfile="${homedir}/samplecell2cndeletion.txt"
cn_amplificationfile="${homedir}/sample/cell2cnamplification.txt"
traindatafile="${homedir}/sample/training_data.txt"
taskfile="${homedir}/sample/task_list_RS.txt"

modeldir="${homedir}/model"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

cudaid=0

pyScript="${homedir}/src/train_helper.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile -train $traindatafile \
	-mutations $mutationfile -cn_deletions $cn_deletionfile -cn_amplifications $cn_amplificationfile \
	-tasks $taskfile -model $modeldir -genotype_hiddens 12 -lr 0.0002 -epoch 300 \
	-batchsize 64 -optimize 1 -cuda $cudaid > "${modeldir}/train.log"
