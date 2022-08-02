#!/bin/bash
#SBATCH --partition=nrnb-compute
#SBATCH --account=nrnb
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --dependency=singleton

cpu_count=8

bash "${1}/scripts/rlipp.sh" $1 $2 $3 $4 $5 $cpu_count

if [ $4 = "CDKi" ]
then
	bash "${1}/scripts/cv_rlipp_genie.sh" $1 $2 $3 $4 $5 $cpu_count
fi
