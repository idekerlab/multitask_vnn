#!/bin/bash
homedir=$1

traindatafile="${homedir}/data/training_files_${3}/train_${3}.txt"
taskfile="${homedir}/data/training_files_${3}/control/task_list_${4}.txt"

modeldir="${homedir}/models/control/model_${2}_${3}_${4}"

pyScript="${homedir}/src/create_train_data.py"
python -u $pyScript -train $traindatafile -tasks $taskfile -model $modeldir -folds $5
