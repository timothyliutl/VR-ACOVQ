#!/bin/bash
#SBATCH --mem=50g
#SBATCH --time=99:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
module load StdEnv/2023
module load cuda
module load python/3.10
cd ../
source venv310/bin/activate 
python vrq_collection.py $SLURM_ARRAY_TASK_ID