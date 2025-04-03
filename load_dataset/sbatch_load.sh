#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=20GB
#SBATCH --time=04:00:00

module load conda/latest
conda activate /project/pi_wenlongzhao_umass_edu/6/sudharshan/envs/lora-finetune

python load_dataset/main.py