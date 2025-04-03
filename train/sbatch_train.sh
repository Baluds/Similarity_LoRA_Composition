#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=l40s:1
#SBATCH --time=8:00:00
#SBATCH --job-name=hindi_math-v1-r64       #Set the job name to "JobName"
#SBATCH --ntasks-per-node=2      #Request 4 tasks/cores per node


if [ $# -eq 0 ]; then
    echo "Usage: sbatch $0 <config_file_path>"
    exit 1
fi

# Print GPU info
echo "### GPU Information ###"
nvidia-smi
echo "#######################"

module load conda/latest
conda activate /project/pi_wenlongzhao_umass_edu/6/sudharshan/envs/lora-finetune

python train/main.py $1