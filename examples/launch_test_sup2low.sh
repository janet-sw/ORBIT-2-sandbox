#!/bin/bash
#SBATCH -A csc662
#SBATCH -J test-sup2low
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -o logs/test-sup2low-%j.out

set -euo pipefail

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

# Load modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.0.0

# Activate conda environment
source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/proj-shared/janet/miniconda3/envs/orbit_main

# Paths - MODIFY THESE
CONFIG_PATH="/ccs/home/janetw/diffusion/ORBIT-2/configs/config_sup2low_forecasting.yaml"
CHECKPOINT_PATH="/lustre/orion/csc662/proj-shared/janet/forecasting/orbit_main/sup2low_forecasting_baseline_lat_mse/checkpoint_epoch_0029.pt"

# Environment variables for ROCm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=7
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
mkdir -p $MIOPEN_USER_DB_PATH

echo "Using configuration: ${CONFIG_PATH}"
echo "Using checkpoint: ${CHECKPOINT_PATH}"

# Run test - single GPU
time python test_sup2low_model.py "${CONFIG_PATH}" "${CHECKPOINT_PATH}"

echo "Job completed at: $(date)"
