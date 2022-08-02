#!/bin/bash

homedir=$1

ontology="${homedir}/data/training_files_${3}/ontology_${2}_${3}.txt"
gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}"
test="${modeldir}/test.txt"

predicted="${modeldir}/predict.txt"
sys_output="${modeldir}/rlipp.out"
gene_output="${modeldir}/gene_rho.out"
hidden="${modeldir}/hidden"

cpu_count=$6

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -test $test \
	-gene_output $gene_output -predicted $predicted -cpu_count $cpu_count  > "${modeldir}/rlipp.log"
