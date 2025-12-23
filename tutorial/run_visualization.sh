#!/bin/bash
#SBATCH -A csc662
#SBATCH -J visualize-superres
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o logs/visualize-%j.out
#SBATCH -e logs/visualize-%j.out

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
export OMP_NUM_THREADS=4
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

############################################################
# Run visualization script
############################################################
echo "="*70
echo "Starting super-resolution visualization"
echo "Time: $(date)"
echo "Working directory: $PWD"
echo "="*70

time python visualize_superres.py

echo ""
echo "="*70
echo "Visualization complete!"
echo "Time: $(date)"
echo "="*70
