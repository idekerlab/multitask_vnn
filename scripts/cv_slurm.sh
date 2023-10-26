#!/bin/bash
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --dependency=singleton

#bash "${1}/scripts/cv_train.sh" "$1" "$2" "$3" "$4" "$5"
if [ "$6" = "TCGA" ]
then
	#bash "${1}/scripts/create_test_input.sh" "$1" "$2" "$3" "$4" "$5" "$6"
	bash "${1}/scripts/cv_test_tcga.sh" "$1" "$2" "$3" "$4" "$5"
else
	if [ "$6" != "" ]
	then
		bash "${1}/scripts/create_test_input.sh" "$1" "$2" "$3" "$4" "$5" "$6"
	fi
	bash "${1}/scripts/cv_test.sh" "$1" "$2" "$3" "$4" "$5" "$6"
fi
