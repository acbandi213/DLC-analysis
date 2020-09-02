#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4
#SBATCH --mail-user=acbandi@princeton.edu
#SBATCH --mail-type=END


cd /scratch/gpfs/acbandi
# from tigergpu
module load anaconda3/5.3.1 cudatoolkit/9.2 cudnn/cuda-9.2/7.3.1
source activate deeplabcut

python analyze_videos.py ${1} ${2}

export ScratchDir="/tmp/test"
mkdir -p $ScratchDir
./a.out $ScratchDir
cp -r $ScratchDir /tigress/acbandi/
