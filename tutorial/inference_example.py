"""
Example script showing how to use the loaded checkpoint for inference.

Usage:
    python inference_example.py --checkpoint /path/to/checkpoint.ckpt --era5_dir /path/to/era5/data
"""

from argparse import ArgumentParser
import torch
from load_checkpoint import load_checkpoint

def run_inference(model, dm, num_samples=5):
    """
    Run inference on a few samples from the test set.
    
    Args:
        model: Loaded model
        dm: Data module
        num_samples: Number of samples to run inference on
    """
    print("\n" + "="*60)
    print("Running inference on test samples...")
    print("="*60)
    
    # Get test dataloader
    test_loader = dm.test_dataloader()
    
    # Get a batch
    batch = next(iter(test_loader))
    inputs, targets, variables, out_variables = batch
    
    print(f"\nBatch information:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Number of input variables: {len(variables)}")
    print(f"  Number of output variables: {len(out_variables)}")
    
    # Move to model's device
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(inputs)
    
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Calculate some basic statistics
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    print(f"\nMetrics on this batch:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    return predictions, targets, inputs


def main():
    parser = ArgumentParser(description="Run inference with loaded checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/lustre/orion/csc662/proj-shared/janet/forecasting/res_slimvit_direct_forecasting_120/checkpoints/epoch_004-v1.ckpt",
        help="Path to the checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--era5_dir",
        type=str,
        required=True,
        help="Path to ERA5 data directory"
    )
    parser.add_argument(
        "--forecast_type",
        type=str,
        default="direct",
        choices=["direct", "iterative", "continuous"],
        help="Type of forecasting"
    )
    parser.add_argument(
        "--pred_range",
        type=int,
        default=120,
        help="Prediction range in hours"
    )
    
    args = parser.parse_args()
    
    # Load the checkpoint
    print("Loading model checkpoint...")
    model, dm = load_checkpoint(
        args.checkpoint,
        args.era5_dir,
        args.forecast_type,
        args.pred_range,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run inference
    predictions, targets, inputs = run_inference(model, dm)
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
    print("\nYou can now use the model for further predictions:")
    print("  - Use model(input_tensor) for forward pass")
    print("  - predictions are in the same format as training")
    print("  - Remember to keep model in eval mode: model.eval()")
    
    return model, dm, predictions, targets, inputs


if __name__ == "__main__":
    model, dm, predictions, targets, inputs = main()
