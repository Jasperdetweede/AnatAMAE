#!/bin/bash

#SBATCH --job-name="anatamae_100_epochs"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py

echo "Starting finetune.py"
srun python finetune.py > finetune.log

echo "Starting pretrain.py"
srun python pretrain.py > pretrain.log

echo "Starting analyze.py"
srun python analyze.py > analyze.log