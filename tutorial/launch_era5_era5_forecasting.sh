#!/bin/bash
#SBATCH -A csc662
#SBATCH -J era5-forecast
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -o logs/sup_forecast-%j.out
#SBATCH -e logs/sup_forecast-%j.out

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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Additional library paths (e.g., RCCL plugin)
export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=Simple
# Per‑job cache for MIOpen kernels
export MIOPEN_USER_DB_PATH=/lustre/orion/csc662/proj-shared/janet/miopen_cache/$SLURM_JOB_ID
export MIOPEN_DISABLE_CACHE=1
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
MAX_EPOCHS=50
PATIENCE=20
# Output directory path
OUTPUT_DIR="/lustre/orion/csc662/proj-shared/janet/forecasting"
CHECKPOINT=""

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
time srun -n $((SLURM_JOB_NUM_NODES * 8)) \
  python ./era5_era5_forecasting.py \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --output_dir "$OUTPUT_DIR" \
    "$FORECAST_TYPE" "$ERA5_DIR" "$MODEL" "$PRED_RANGE"