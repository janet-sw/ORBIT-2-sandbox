#!/bin/bash

# Exit on error, unset variables and pipefail
set -euo pipefail

############################################################
# Modules & Conda environment
############################################################
module purge
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.0

# Activate the appropriate conda environment for ORBIT‑2/ClimateLearn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/janetw/miniconda3/envs/orbit

############################################################
# Runtime environment variables
############################################################
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=7
# Additional library paths (e.g., RCCL plugin)
export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
# Per‑job cache for MIOpen kernels
export MIOPEN_USER_DB_PATH=$PWD/miopen_cache/$$
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

mkdir -p "$MIOPEN_USER_DB_PATH"

############################################################
# Paths & run configuration
############################################################
# Input directory path
ERA5_DIR="/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/"
# Forecast type (direct, iterative or continuous)
FORECAST_TYPE="direct"
# Deep learning backbone – use res_slimvit to leverage Reslim architecture
MODEL="res_slimvit"
# Prediction range in hours
PRED_RANGE=120
# Training hyper‑parameters
MAX_EPOCHS=5
PATIENCE=5
# Output directory path
OUTPUT_DIR="/lustre/orion/csc662/proj-shared/janet/forecasting"
CHECKPOINT="/lustre/orion/csc662/proj-shared/janet/forecasting/res_slimvit_direct_forecasting_120/checkpoints/epoch_004.ckpt"

# Number of GPUs to use (default: 8 for local run)
NUM_GPUS=8

echo "Job configuration:"
echo "  ERA5_DIR      = ${ERA5_DIR}"
echo "  FORECAST_TYPE = ${FORECAST_TYPE}"
echo "  MODEL         = ${MODEL}"
echo "  PRED_RANGE    = ${PRED_RANGE}"
echo "  MAX_EPOCHS    = ${MAX_EPOCHS}"
echo "  PATIENCE      = ${PATIENCE}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  CHECKPOINT    = ${CHECKPOINT}"
echo "  NUM_GPUS      = ${NUM_GPUS}"

############################################################
# Launch the forecasting job across all available GPUs
############################################################
# Run within existing interactive allocation
srun -N1 -n8 --gpus-per-task=1 --gpu-bind=closest time python ./era5_era5_forecasting.py \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --output_dir "$OUTPUT_DIR" \
    "$FORECAST_TYPE" "$ERA5_DIR" "$MODEL" "$PRED_RANGE"
