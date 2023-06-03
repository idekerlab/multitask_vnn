#!/bin/bash

homedir="your_home_directory"

gene2idfile="${homedir}/sample/gene2ind.txt"
cell2idfile="${homedir}/sample/cell2ind.txt"
mutationfile="${homedir}/sample/cell2mutation.txt"
testdatafile="${homedir}/sample/test_data.txt"

modeldir="${homedir}/sample/model"
modelfile="${modeldir}/model_final.pt"

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

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile \
	-genotype $mutationfile -hidden $hiddendir -result $resultfile \
	-batchsize 2000 -predict $testdatafile -load $modelfile -cuda $cudaid > "${modeldir}/test.log"
