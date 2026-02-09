#!/bin/bash
#SBATCH -A CSC662
#SBATCH -J era5_forecast_test
#SBATCH -o logs/era5_forecast_test-%j.out
#SBATCH -e logs/era5_forecast_test-%j.out
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7

# Load necessary modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.3.1

# Activate your environment
source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate orbit_main

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for ROCm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=7
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
mkdir -p $MIOPEN_USER_DB_PATH

export PYTHONPATH=$PWD/../src:$PYTHONPATH
export ORBIT_USE_DDSTORE=0

# Configuration and checkpoint paths
CONFIG_PATH="/ccs/home/janetw/diffusion/ORBIT-2-sandbox/configs/era5_forecasting.yaml"
CHECKPOINT_PATH="/lustre/orion/csc662/proj-shared/janet/forecasting/orbit_main/era5_forecasting_1.40625_deg_bilinear_antialias=True_input_refine_cnn=True/checkpoint_epoch_0029.pt"

echo "=========================================="
echo "Testing ERA5 Forecasting Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================================="
echo ""

# Run the test script - single GPU
time python test_era5_forecasting.py \
    $CONFIG_PATH \
    $CHECKPOINT_PATH

echo ""
echo "=========================================="
echo "Testing complete!"
echo "=========================================="
