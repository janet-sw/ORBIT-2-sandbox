#!/bin/bash
#SBATCH -A CSC662
#SBATCH -J era5_forecast
#SBATCH -o logs/era5_forecast-%j.out
#SBATCH -e logs/era5_forecast-%j.out
#SBATCH -t 00:40:00
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7

# Exit on error
set -euo pipefail

############################################################
# Modules & Conda environment
############################################################
module purge
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.0.0

source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/proj-shared/janet/miniconda3/envs/orbit_main

############################################################
# Runtime environment variables
############################################################
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/lustre/orion/csc662/proj-shared/janet/miopen_cache/$SLURM_JOB_ID
export MIOPEN_DISABLE_CACHE=1
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

mkdir -p "$MIOPEN_USER_DB_PATH"
mkdir -p logs

############################################################
# Run configuration — matching res_slimvit 1.40625° setup
############################################################
# Data: ERA5 1.40625° (128x256 native grid)
ERA5_DIR="/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/"

FORECAST_TYPE="direct"
MODEL="vit"
PRED_RANGE=120
MAX_EPOCHS=30
PATIENCE=20

# Output directory — clearly labeled as ViT baseline
OUTPUT_DIR="/lustre/orion/csc662/proj-shared/janet/forecasting/vit_baseline_1.40625_deg"

echo "============================================================"
echo "ViT Baseline — ERA5 1.40625° (matched to res_slimvit config)"
echo "============================================================"
echo "  ERA5_DIR      = ${ERA5_DIR}"
echo "  MODEL         = ${MODEL}"
echo "  PRED_RANGE    = ${PRED_RANGE}"
echo "  MAX_EPOCHS    = ${MAX_EPOCHS}"
echo "  NODES         = ${SLURM_JOB_NUM_NODES}"
echo "  TOTAL GPUs    = $((SLURM_JOB_NUM_NODES * 8))"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "============================================================"

############################################################
# Launch — 2 nodes x 8 GPUs = 16 ranks (same as res_slimvit)
############################################################
time srun -n $((SLURM_JOB_NUM_NODES * 8)) \
  python ./era5_era5_forecasting.py \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --output_dir "$OUTPUT_DIR" \
    --img_size 128 256 \
    --history 1 \
    --batch_size 16 \
    --patch_size 2 \
    --superres_factor 1 \
    "$FORECAST_TYPE" "$ERA5_DIR" "$MODEL" "$PRED_RANGE"