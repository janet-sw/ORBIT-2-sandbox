#!/bin/bash
#SBATCH -A CSC662
#SBATCH -J sup2low_forecast
#SBATCH -o logs/sup2low-%j.out
#SBATCH -e logs/sup2low-%j.out
#SBATCH -t 00:40:00
#SBATCH -q debug
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.0.0

# Activate conda environment - use the correct conda.sh path
source /lustre/orion/csc662/proj-shared/janet/miniconda3/etc/profile.d/conda.sh
conda activate orbit_main

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

# Set environment variables
# MIOpen cache settings - use local /tmp to avoid SQLite disk I/O errors on Lustre
export MIOPEN_USER_DB_PATH=/tmp/miopen_$USER_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_$SLURM_JOB_ID
mkdir -p $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_CUSTOM_CACHE_DIR
export OMP_NUM_THREADS=7

# NCCL/RCCL debugging and stability settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# Disable ROCm SMI to avoid segfault in rsmi_dev_unique_id_get
export ROCM_SMI_DISABLE=1
export HSA_ENABLE_SDMA=0

# Additional workarounds for ROCm SMI crashes
export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=4
export NCCL_IGNORE_CPU_AFFINITY=1

# Prevent RCCL from querying GPU PCI info
export RCCL_MSCCL_ENABLE=0
export NCCL_TREE_THRESHOLD=0

# Ultimate workaround: Force simpler network transport to avoid PCI queries
export NCCL_NET="Socket"
export NCCL_P2P_DISABLE=1

# Configuration file
CONFIG_FILE="/ccs/home/janetw/diffusion/ORBIT-2/configs/config_sup2low_forecasting.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"
echo ""

# CRITICAL: Set ROCR_VISIBLE_DEVICES per rank to prevent ROCm SMI crashes
# This ensures each process only sees its own GPU
export SLURM_GPU_DIRECT=1

# Run training with per-rank GPU isolation
time srun -u --export=ALL,ROCR_VISIBLE_DEVICES='$SLURM_LOCALID' python sup2low_forecasting_no_lightning.py "$CONFIG_FILE"

# Print completion time
echo ""
echo "Job completed at: $(date)"
