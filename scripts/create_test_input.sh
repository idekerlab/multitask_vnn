#!/bin/bash
homedir=$1

testdatafile="${homedir}/data/training_files_${3}/train_${3}.txt"
taskfile="${homedir}/data/training_files_${3}/task_list_${6}.txt"

modeldir="${homedir}/models/Final/model_${2}_${3}_${4}_${5}"

pyScript="${homedir}/src/create_multitask_data.py"
python -u $pyScript -data $testdatafile -tasks $taskfile -model $modeldir -mode $6
