#!/bin/bash
#SBATCH -J pi3_pose_extract
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1 -C gmem32
#SBATCH --output=runs/pose_extract_%j.out
# mkdir -p runs   # run once before first submission -- SLURM needs the dir to already exist

module load anaconda3
module load cuda/11.4

export PATH="/home/de575594/.conda/envs/pi3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate pi3

# --- Edit these before submitting ---
VIDEO_PATH="videos/Piran_Walking_Tour_cut.mp4"
OUTPUT_DIR="output_videos/Piran_v1"

# Skips Pi3's own walk-vs-pan sanity check (--skip-calibration) since
# validation happens downstream on your own system -- goes straight to the
# full video. Remove --skip-calibration (see pi3_pose_calibration.sh) if you
# ever want that pre-flight check back.
python scripts/extract_camera_pose.py "$VIDEO_PATH" "$OUTPUT_DIR" \
    --model pi3x \
    --batch-size 100 \
    --overlap 12 \
    --skip-calibration
