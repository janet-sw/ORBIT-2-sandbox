

# Activate the appropriate conda environment for ORBIT‑2/ClimateLearn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ccs/home/janetw/miniconda3/envs/orbit


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
python ./era5_era5_forecasting.py \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --output_dir "$OUTPUT_DIR" \
    "$FORECAST_TYPE" "$ERA5_DIR" "$MODEL" "$PRED_RANGE"