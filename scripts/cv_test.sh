#!/bin/bash
homedir=$1

gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"
mutationfile="${homedir}/data/training_files_${3}/cell2mutation_${2}_${3}.txt"
cn_deletionfile="${homedir}/data/training_files_${3}/cell2cndeletion_${2}_${3}.txt"
cn_amplificationfile="${homedir}/data/training_files_${3}/cell2cnamplification_${2}_${3}.txt"

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}"
modelfile="${modeldir}/model_final.pt"
testdatafile="${modeldir}/test.txt"

resultfile="${modeldir}/predict"

hiddendir="${modeldir}/hidden"
if [ -d $hiddendir ]
then
	rm -rf $hiddendir
fi
mkdir -p $hiddendir

cudaid=0

pyScript="${homedir}/src/test_vnn.py"

source activate cuda11_env

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile -hidden $hiddendir -result $resultfile \
	-mutations $mutationfile -cn_deletions $cn_deletionfile -cn_amplifications $cn_amplificationfile \
	-batchsize 2000 -predict $testdatafile -load $modelfile -cuda $cudaid > "${modeldir}/test.log"
