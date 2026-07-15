#!/bin/bash
#SBATCH -J pi3_pose_calib
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem12
#SBATCH --output=runs/pose_calib_%j.out
# mkdir -p runs   # run once before first submission -- SLURM needs the dir to already exist

module load anaconda3
module load cuda/11.4

export PATH="/home/de575594/.conda/envs/pi3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate pi3

# --- Edit these before submitting ---
VIDEO_PATH="/path/to/your/walking_tour.mp4"
OUTPUT_DIR="/path/to/output_dir"

# Cheap sanity check on a short clip before committing to the full video.
# Point --calibration-walk-range / --calibration-still-range at seconds
# (cut-video time) you know are real walking / standing-still-while-panning
# to get an automatic PASS/FAIL verdict instead of INCONCLUSIVE.
python scripts/extract_camera_pose.py "$VIDEO_PATH" "$OUTPUT_DIR" \
    --model pi3x \
    --calibration-only \
    --calibration-start 0 \
    --calibration-duration 240 \
    --calibration-walk-range 30-90 \
    --calibration-still-range 100-160
