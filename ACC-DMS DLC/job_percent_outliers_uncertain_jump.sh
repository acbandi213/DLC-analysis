#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-job_percent_outliers_uncertain_jump-%j.out
#SBATCH --mail-user=acbandi@princeton.edu
#SBATCH --mail-type=END

cd $(pwd)

python percent_outliers_uncertain_jump.py ${1}