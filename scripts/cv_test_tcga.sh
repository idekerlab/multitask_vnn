#!/bin/bash
homedir=$1

gene2idfile="${homedir}/data/TCGA/gene2ind.txt"
cell2idfile="${homedir}/data/TCGA/dichloroplatinum-diammoniate/cell2ind.txt"
mutationfile="${homedir}/data/TCGA/dichloroplatinum-diammoniate/cell2mutation.txt"
cn_deletionfile="${homedir}/data/TCGA/dichloroplatinum-diammoniate/cell2cndeletion.txt"
cn_amplificationfile="${homedir}/data/TCGA/dichloroplatinum-diammoniate/cell2cnamplification.txt"

modeldir="${homedir}/models/Final/model_${2}_${3}_${4}_${5}"
modelfile="${modeldir}/model_final.pt"
testdatafile="${modeldir}/test_tcga.txt"

resultfile="${modeldir}/predict_tcga"

hiddendir="${modeldir}/hidden_tcga"
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
