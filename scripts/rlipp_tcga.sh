#!/bin/bash

homedir=$1

ontology="${homedir}/data/training_files_${3}/ontology_${2}_${3}.txt"
gene2idfile="${homedir}/data/TCGA/gene2ind.txt"
cell2idfile="${homedir}/data/TCGA/dichloroplatinum-diammoniate/cell2ind.txt"

modeldir="${homedir}/models/Final/model_${2}_${3}_${4}_${5}"
test="${modeldir}/test_tcga.txt"

predicted="${modeldir}/predict_tcga.txt"
sys_output="${modeldir}/rlipp_tcga.out"
gene_output="${modeldir}/gene_rho_tcga.out"
hidden="${modeldir}/hidden_tcga"

cpu_count=$6

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -test $test \
	-gene_output $gene_output -predicted $predicted -cpu_count $cpu_count  > "${modeldir}/rlipp.log"

