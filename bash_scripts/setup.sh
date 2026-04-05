#!/bin/bash
#SBATCH -J pi3_setup
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem12
#SBATCH --output=runs/setup_job.out

module load anaconda3
module load cuda/11.4

export PATH="/home/de575594/.conda/envs/pi3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate pi3

pip install -r requirements.txt