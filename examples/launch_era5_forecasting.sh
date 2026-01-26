#!/bin/bash
#SBATCH -A CSC662
#SBATCH -J era5_forecast
#SBATCH -o logs/era5_forecast-%j.out
#SBATCH -e logs/era5_forecast-%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.3.1

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
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_DISABLE_CACHE=1
export OMP_NUM_THREADS=7
export NCCL_PROTO=Simple
export PYTHONNOUSERSITE=1
export MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX=-1
export MIOPEN_DEBUG_CONV_WINOGRAD=0

# NCCL/RCCL debugging and stability settings
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.

# Prevent RCCL from querying GPU PCI info
export RCCL_MSCCL_ENABLE=0
export NCCL_TREE_THRESHOLD=0

export PYTHONPATH=$PWD/../src:$PYTHONPATH
export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)


# Configuration file
CONFIG_FILE="/ccs/home/janetw/diffusion/ORBIT-2-sandbox/configs/era5_forecasting.yaml"

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python era5_forecasting.py "$CONFIG_FILE"