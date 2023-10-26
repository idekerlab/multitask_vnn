#!/bin/bash
homedir=$1

mode="train"

traindatafile="${homedir}/data/training_files_${3}/train_${3}.txt"
taskfile="${homedir}/data/training_files_${3}/control/task_list_${4}.txt"

modeldir="${homedir}/models/control/model_${2}_${3}_${4}"

pyScript="${homedir}/src/create_multitask_data.py"
python -u $pyScript -data $traindatafile -tasks $taskfile -model $modeldir -folds $5 -mode "train"
