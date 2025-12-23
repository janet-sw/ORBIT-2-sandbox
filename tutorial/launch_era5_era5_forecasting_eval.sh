#!/bin/bash
#SBATCH -A csc662
#SBATCH -J era5-forecast
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -o logs/forecast-%j.out
#SBATCH -e logs/forecast-%j.out

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
source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/proj-shared/janet/miniconda3/envs/orbit

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
export MIOPEN_USER_DB_PATH=$PWD/miopen_cache/$SLURM_JOB_ID
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

mkdir -p "$MIOPEN_USER_DB_PATH"

############################################################
# Paths & run configuration
############################################################
# Input directory path
ERA5_DIR="/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/"
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
CHECKPOINT="/lustre/orion/csc662/proj-shared/janet/forecasting/v4_test_trans_resolution_res_slimvit_direct_forecasting_120/checkpoints/last.ckpt"

echo "Job configuration:"
echo "  ERA5_DIR      = ${ERA5_DIR}"
echo "  FORECAST_TYPE = ${FORECAST_TYPE}"
echo "  MODEL         = ${MODEL}"
echo "  PRED_RANGE    = ${PRED_RANGE}"
echo "  MAX_EPOCHS    = ${MAX_EPOCHS}"
echo "  PATIENCE      = ${PATIENCE}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  CHECKPOINT    = ${CHECKPOINT}"

############################################################
# Launch the forecasting job across all available GPUs
############################################################
# Each MPI rank (one per GPU) calls era5_era5_forecasting.py.  The script
# uses SLURM_PROCID and SLURM_LOCALID internally to select the correct device.
time srun -n $((SLURM_JOB_NUM_NODES)) \
  python ./era5_era5_forecasting_eval.py \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --img_size 32 64 \
    --superres_factor 4 \
    --downsample_mode area \
    "$FORECAST_TYPE" "$ERA5_DIR" "$MODEL" "$PRED_RANGE"