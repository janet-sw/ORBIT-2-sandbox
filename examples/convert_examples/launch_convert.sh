#!/bin/bash
#SBATCH -A csc662
#SBATCH -J convert_era5
#SBATCH -o convert_era5-%j.out
#SBATCH -e convert_era5-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -p batch

# ─── Paths ───
INPUT_DIR="/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-1hr-superres/1.0_deg/"
OUTPUT_DIR="/lustre/orion/world-shared/csc662/janetw/era5_1.0_deg_timesteps/"
SCRIPT_DIR="/ccs/home/janetw/diffusion/ORBIT-2-sandbox"

# ─── Environment ───
module load PrgEnv-gnu
module load cray-python

cd ${SCRIPT_DIR}

echo "============================================"
echo "ERA5 Conversion Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Quick sanity check
echo ""
echo "Input directory contents:"
ls ${INPUT_DIR}/ | head -5
echo ""
echo "Test files:"
ls ${INPUT_DIR}/test/*.npz 2>/dev/null | wc -l
echo ""

# Run conversion with 16 parallel workers
python /ccs/home/janetw/diffusion/ORBIT-2-sandbox/examples/convert_era5_timesteps.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 16

echo ""
echo "============================================"
echo "Conversion finished at $(date)"
echo "Output size:"
du -sh ${OUTPUT_DIR}
echo ""
echo "Test timesteps:"
ls ${OUTPUT_DIR}/test/*.npy 2>/dev/null | wc -l
echo "============================================"