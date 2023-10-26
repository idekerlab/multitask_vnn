#!/bin/bash

homedir=$1

ontology="${homedir}/data/training_files_${3}/ontology_${2}_${3}.txt"
gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"

modeldir="${homedir}/models/Final/model_${2}_${3}_${4}_${5}"

test="${modeldir}/test.txt"
predicted="${modeldir}/predict.txt"
hidden="${modeldir}/hidden"
sys_output="${modeldir}/rlipp.out"
gene_output="${modeldir}/gene_rho.out"

if [ "$6" != "" ]; then
	test="${modeldir}/${6}.txt"
	predicted="${modeldir}/predict_${6}.txt"
	sys_output="${modeldir}/rlipp_${6}.out"
	gene_output="${modeldir}/gene_rho_${6}.out"
	hidden="${modeldir}/hidden_${6}"
fi

cpu_count=$7

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -test $test \
	-gene_output $gene_output -predicted $predicted -cpu_count $cpu_count  > "${modeldir}/rlipp.log"
