#!/bin/bash
#SBATCH --mem=30g
#SBATCH --time=95:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
module load StdEnv/2020
module load cuda
module load python/3.10
cd ../
source venv310/bin/activate 
python data_collection_tsvq_acovq.py $SLURM_ARRAY_TASK_ID