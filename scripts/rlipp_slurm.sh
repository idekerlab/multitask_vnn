#!/bin/bash
#SBATCH --partition=nrnb-compute
#SBATCH --account=nrnb
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --dependency=singleton

cpu_count=8

if [ "$6" = "TCGA" ]; then
	bash "${1}/scripts/rlipp_tcga.sh" "$1" "$2" "$3" "$4" "$5" "$cpu_count"
else
	bash "${1}/scripts/rlipp.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$cpu_count"
fi