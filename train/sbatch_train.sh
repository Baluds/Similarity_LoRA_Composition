#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name=IMDB-v1-r64       #Set the job name to "JobName"
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 40GB



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