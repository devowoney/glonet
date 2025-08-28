import torch
import torch.nn.functional as F
import sys
import xarray as xr
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from optimInitLit import Glonet, GlorysDataModule

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig):
    # Configuration for multiple ocean states
    state_configs = {
        1: {'model_file': 'glonet_p1.pt', 'input_file': 'input1.nc'},
        2: {'model_file': 'glonet_p2.pt', 'input_file': 'input2.nc'},
        3: {'model_file': 'glonet_p3.pt', 'input_file': 'input3.nc'}
    }
    
    model_dir = cfg.model_dir
    data_dir = cfg.data_dir 
    state_number = cfg.state_number 
    sample_index = cfg.sample_index 
    
    if state_number not in state_configs:
        raise ValueError(f"Invalid state number: {state_number}. Choose 1, 2, or 3.")
    
    config = state_configs[state_number]
    model_path = model_dir + "/" + config['model_file']
    input_file = config['input_file']
    
    print(f"=== Optimizing Ocean State {state_number} ===")
    print(f"Model: {model_path}")
    print(f"Input file: {input_file}")
    print(f"Sample index: {sample_index}")
    
    # Initialize data module first to get the actual input sequence
    data_module = GlorysDataModule(
        data_dir=data_dir, 
        input_file=input_file,
        sample_index=sample_index, 
        forecast_horizon=cfg.training.datamodule.forecast_horizon,
        batch_size=cfg.training.datamodule.batch_size, 
        num_workers=cfg.training.datamodule.num_workers
    )
    data_module.setup()

    # Get the input sequence and target for this sample
    input_sequence, target = data_module.train_dataset.input_sequence, data_module.train_dataset.target
    date = data_module.train_dataset.selected_date
    print(f"[Debug]: Selected date for sample {sample_index}: {date}")

    # Initialize model with the actual input sequence as the trainable parameter
    model = Glonet(model_path=model_path, cfg=cfg)
    
    # Move model and data to the correct device to avoid dtype/device mismatches
    device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(cfg.training.trainer, "devices", None) else "cpu")
    model.to(device)
    try:
        model.saved_model = model.saved_model.to(device)
    except Exception:
        pass
    input_sequence = input_sequence.to(device)
    target = target.to(device)
    
    # Reset gradients
    model.zero_grad()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.training.callbacks.monitor,  # Monitor epoch-level train loss
            dirpath=cfg.training.callbacks.dirpath,
            filename=cfg.training.callbacks.filename,
            save_top_k=cfg.training.callbacks.save_top_k,
            mode=cfg.training.callbacks.mode,
            save_last=True,
            # every_n_epochs=10,  # Save every epoch
            verbose=True  # Print when checkpoints are saved
        )
    ]
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir = cfg.training.tensorboard.save_dir,
        name=cfg.training.tensorboard.name,
        version=cfg.training.tensorboard.version
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.training.trainer.epochs,
        callbacks=callbacks,
        accelerator=cfg.training.trainer.accelerator,
        devices=cfg.training.trainer.devices,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        # gradient_clip_val=cfg.training.trainer.grad_clip_norm,
        precision=cfg.training.trainer.precision,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger
    )
    
    # Start training
    trainer.fit(model, data_module)
    
    # Save the optimized input sequence
    print(f"[Debug]: Saving optimized input sequence in NetCDF format")
    # Get best_init (standardized) - numpy
    best_init_std = model.best_init         # shape: (time, channel, lat, lon)

    # Get dataset object to read coords & mean/std
    ds_obj = trainer.datamodule.train_dataset     # OptimizeInitialConditionDataset instance
    orig_xr = ds_obj.dataset                       # original xarray.Dataset opened in the dataset
    mean = ds_obj.mean.cpu().numpy()               # shape (1, channels, 1, 1)
    std  = ds_obj.std.cpu().numpy()

    # Destandardize (broadcasting)
    best_init_unstd = best_init_std * std + mean   # still numpy, same shape

    # Get coordinates (time slice for the input sequence)
    start_idx = ds_obj.valid_indices[ds_obj.sample_index]
    seq_len = best_init_unstd.shape[0]  # Get actual sequence length from data
    time_coords = orig_xr.time.isel(time=slice(start_idx, start_idx + seq_len))
    # spatial coords (handle both names)
    lat = orig_xr.coords.get('lat', orig_xr.coords.get('latitude'))
    lon = orig_xr.coords.get('lon', orig_xr.coords.get('longitude'))

    # Create xarray Dataset and save
    # dims expected: ('time','channel','lat','lon')
    data_array = xr.DataArray(
        best_init_unstd,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": time_coords, 
            "lat": lat, 
            "lon": lon, 
            "channel": np.arange(best_init_unstd.shape[1])
        },
        name="data"
    )
    ds_out = data_array.to_dataset()
    ds_out.attrs["description"] = "Optimized initial condition (destandardized)"
    windows = cfg.training.datamodule.forecast_horizon
    out_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ds_out.to_netcdf(f"{out_path}/optimized_init_input{state_number}_{date}_{windows}_days_window.nc")
    print(f"Saved: Optimized input at {out_path}/optimized_init_input{state_number}_{date}_{windows}_days_window.nc")


    # Calculate and show the improvement
    optimized_input = torch.from_numpy(model.best_init).cpu()  # Convert numpy to tensor
    original_input = input_sequence.detach().cpu()
    input_diff = torch.abs(optimized_input - original_input).mean()
    
    print(f"\n=== Optimization Results for State {state_number} ===")
    print(f"Model: {model_path}")
    print(f"Input file: {input_file}")
    print(f"Sample index: {sample_index}")
    print(f"Input shape: {optimized_input.shape}")
    print(f"Original input stats: mean={original_input.mean():.4f}, std={original_input.std():.4f}")
    print(f"Optimized input stats: mean={optimized_input.mean():.4f}, std={optimized_input.std():.4f}")
    print(f"Mean absolute change: {input_diff:.6f}")
    print(f"Final loss: {trainer.logged_metrics.get('train_loss_epoch', 'N/A')}")
    
    # Test the optimized input vs original
    model.eval()
    with torch.no_grad():
        # Ensure tensors and model are on same device
        original_input_device = original_input.to(device).unsqueeze(0)
        optimized_input_device = optimized_input.to(device).unsqueeze(0)  # Add batch dimension
        target_device = target.to(device).unsqueeze(0)  # Add batch dimension for target too
        try:
            model.saved_model = model.saved_model.to(device)
        except Exception:
            pass
        print(f"[Debug] : model device = {next(model.saved_model.parameters()).device}")
        print(f"[Debug] : original_input_device device = {original_input_device.device}")
        print(f"[Debug] : optimized_input_device device = {optimized_input_device.device}")
        print(f"[Debug] : target_device device = {target_device.device}")
        
        original_prediction, original_prediction_nc = model.forecast(original_input_device)
        optimized_prediction, optimized_prediction_nc = model.forecast(optimized_input_device)

        original_loss = F.mse_loss(original_prediction, target_device)
        optimized_loss = F.mse_loss(optimized_prediction, target_device)
        
        improvement = (original_loss - optimized_loss).item()
        improvement_percent = (improvement / original_loss.item()) * 100
        
        print(f"Original input loss: {original_loss.item():.6f}")
        print(f"Optimized input loss: {optimized_loss.item():.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
    
    return model, optimized_input, original_input, target


if __name__ == "__main__":
    main()