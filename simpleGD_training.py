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

from optimInitLit import GlorysDataModule
from glonet import Glonet
from glonet_daily_forecast_local.forecast import aforecast, aforecast2, aforecast3

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
    
    
    # =============================================================================================== Data
    # ====================================================================================================
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
    mean, std = data_module.train_dataset.mean, data_module.train_dataset.std
    date = data_module.train_dataset.selected_date
    land_mask = data_module.train_dataset.land_mask
    print(f"[Debug]: Selected date for sample {sample_index}: {date}")
    
    # Move model and data to the correct device to avoid dtype/device mismatches
    device = torch.device("cpu")  # Default to CPU
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Using CUDA device.")
        else:
            print("CUDA not available; using CPU.")
    except RuntimeError as e:
        print(f"CUDA error detected: {e}. Using CPU.")
    
    
    
    
    # =============================================================================================== Model
    # =====================================================================================================
    # Initialize model with the actual input sequence as the trainable parameter
    model = torch.jit.load(model_path).to(device)
    src_model = Glonet(dim=(2, 5, 672, 1440)).to(device) # <========================== TODO : Change ch from input 
    print(f"[Debug]: Loaded model from {model_path} and {src_model} from < ./src >")
    
    # Copy weights from saved glonet to original source model
    # collect source params/buffers
    src_params = {n: p for n, p in model.named_parameters()} if hasattr(model, 'named_parameters') else {}
    src_bufs   = {n: b for n, b in model.named_buffers()}    if hasattr(model, 'named_buffers')    else {}

    copied = []
    with torch.no_grad():
        for name, tgt in src_model.named_parameters():
            src = src_params.get(name)
            if src is None:
                continue
            if tuple(src.shape) == tuple(tgt.shape):
                tgt.copy_(src.to(tgt.device).to(tgt.dtype))
                copied.append(name)

        for name, tgt in src_model.named_buffers():
            src = src_bufs.get(name)
            if src is None:
                continue
            if tuple(src.shape) == tuple(tgt.shape):
                tgt.copy_(src.to(tgt.device).to(tgt.dtype))

    print("params copied by name:", len(copied))
    
    # build lists of remaining src/tgt tensors
    remaining_src = [p for n, p in src_params.items() if n not in set(copied)]
    remaining_tgt = [(n, p) for n, p in src_model.named_parameters() if n not in set(copied)]

    with torch.no_grad():
        for tgt_name, tgt in remaining_tgt:
            for i, src in enumerate(remaining_src):
                if tuple(src.shape) == tuple(tgt.shape):
                    tgt.copy_(src.to(tgt.device).to(tgt.dtype))
                    print(f"copied by shape: {tgt_name} <- src_index_{i} shape={src.shape}")
                    remaining_src.pop(i)
                    break
    
    # Confirm data and model are in gpu
    try:
        src_model.to(device)
        input_sequence = input_sequence.unsqueeze(0).to(device)
        target = target.to(device)
        mean = mean.to(device)
        std = std.to(device)
        land_mask = land_mask.to(device)
    except RuntimeError as e:
        print(f"Error moving to device: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        src_model.to(device)
        input_sequence = input_sequence.unsqueeze(0).to(device)
        target = target.to(device)
        mean = mean.to(device)
        std = std.to(device)
        land_mask = land_mask.to(device)
    
    # Delete loaded model for gpu memory
    model.to('cpu')
    torch.cuda.empty_cache()



    # =============================================================================================== Training
    # ========================================================================================================
    # Freeze model parameters
    for p in src_model.parameters():
        p.requires_grad = False
    src_model.eval()

    # input to param for gradient flow
    x_init = input_sequence.detach().clone().to(device)
    x_param = torch.nn.Parameter(x_init, requires_grad=True)
    
    # Expand land_mask to match x_param dimensions (batch, time, channel, lat, lon)
    # land_mask has shape (channel, lat, lon), x_param has shape (batch, time, channel, lat, lon)
    expanded_mask = land_mask.unsqueeze(0).unsqueeze(0).expand_as(x_param)
    print(f"[Debug]: x_param shape: {x_param.shape}")
    print(f"[Debug]: land_mask shape: {land_mask.shape}")
    print(f"[Debug]: expanded_mask shape: {expanded_mask.shape}")
    print(f"[Debug]: Ocean points in mask: {expanded_mask.sum().item()}")

    # Define forecast function for multiple steps (retrorepetive)
    def gloent_forecast(x : torch.Tensor) -> torch.Tensor:
        
        for i in range(cfg.training.datamodule.forecast_horizon):
            glo_out = src_model(x.to(device))
            glo_out = glo_out * expanded_mask
            x = glo_out
        glo_out.to("cpu")
        y = x.clone()
        x.to("cpu")
        torch.cuda.empty_cache()
        return y
    
    # Define learning rate for manual update
    learning_rate = 1e7
    momentum = 0.7
    velocity = torch.zeros_like(x_param)  # For momentum
    
    # Reset gradients
    src_model.zero_grad()

    start_norm = x_param.data.norm().item()
    print(f'start x norm: {start_norm:.6f}')

    # small training loop with manual masked gradient updates
    steps = 100
    for i in range(steps):
        # Zero gradients manually
        if x_param.grad is not None:
            x_param.grad.zero_()
        
        out = gloent_forecast(x_param)
        
        loss = torch.nn.functional.mse_loss(out, target)  # normalized loss
        loss.backward()

        # Manual gradient update with land mask
        with torch.no_grad():
            if x_param.grad is not None:
                # Apply mask to gradients: only ocean points (mask=1) get updated
                masked_grad = expanded_mask * x_param.grad
                
                # Apply momentum
                velocity = momentum * velocity + masked_grad
                
                # Manual parameter update: x_new = x_old - learning_rate * (mask * grad)
                x_param.data = x_param.data - learning_rate * velocity

        if i % 100 == 0 or i == steps - 1:
            grad_norm = x_param.grad.norm().item() if x_param.grad is not None else float('nan')
            masked_grad_norm = (expanded_mask * x_param.grad).norm().item() if x_param.grad is not None else float('nan')
            print(f'step {i:02d} loss={loss.item():.6e} full_grad_norm={grad_norm:.6e} '
                  f'masked_grad_norm={masked_grad_norm:.6e} x.norm={x_param.data.norm().item():.6f}')

    end_norm = x_param.data.norm().item()
    print(f'end x norm: {end_norm:.6f} (changed by {end_norm - start_norm:.6f})')
    print('final x_param.requires_grad:', x_param.requires_grad)
    print('out_t.requires_grad:', getattr(out, 'requires_grad', None), 'out_t.grad_fn:', getattr(out, 'grad_fn', None))
    
    # Destandardize and move to numpy
    best_array = (x_param.detach().clone().cpu().numpy() * std.cpu().numpy()) + mean.cpu().numpy()

    # If a batch dimension exists (we unsqueezed earlier), remove it so dims are (time, channel, lat, lon)
    if best_array.ndim == 5:
        best_array = best_array.squeeze(0)

    # Retrieve coordinates from the original xarray dataset stored on the dataset object
    ds = data_module.train_dataset.dataset
    seq_len = getattr(data_module.train_dataset, 'sequence_length', None)
    target_idx = getattr(data_module.train_dataset, 'target_idx', None)

    if seq_len is not None and target_idx is not None:
        # start_idx used when building the sample in the dataset
        start_idx = int(target_idx - seq_len - data_module.train_dataset.forecast_horizon + 1)
        time_coords = ds.time.isel(time=slice(start_idx, start_idx + seq_len))
    else:
        print("[Warning]: sequence_length or target_idx not found in dataset attributes.")

    # spatial coords (handle both names)
    lat = ds.coords.get('lat', ds.coords.get('latitude'))
    lon = ds.coords.get('lon', ds.coords.get('longitude'))

    # Create xarray Dataset and save
    # dims expected: ('time','channel','lat','lon')
    data_array = xr.DataArray(
        best_array,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon,
            "channel": np.arange(best_array.shape[1])
        },
        name="data"
    )
    ds_out = data_array.to_dataset()
    ds_out.attrs["description"] = "Optimized initial condition (destandardized)"
    windows = cfg.training.datamodule.forecast_horizon
    ds_out.to_netcdf(f"SGDoptimized_init_input{state_number}_{date}_{windows}_days_window.nc")
    
    print(f"Saved: Optimized input at ./SGDoptimized_init_input{state_number}_{date}_{windows}_days_window.nc")

    # ============================================================================================= Validation
    # ========================================================================================================

    y = (target.detach().clone().cpu().numpy() * std.cpu().numpy()) + mean.cpu().numpy()
    print(f"[Debug]: y shape: {y.shape}")
    y_hat = (gloent_forecast(x_init).detach().clone().cpu().numpy() * std.cpu().numpy()) + mean.cpu().numpy()
    y_hat = y_hat.squeeze(0)  # Remove batch dim if exists
    print(f"[Debug]: y_hat shape: {y_hat.shape}")
    optimized_forecast = gloent_forecast(x_param)
    y_tilda_hat = (optimized_forecast.detach().clone().cpu().numpy() * std.cpu().numpy()) + mean.cpu().numpy()
    y_tilda_hat = y_tilda_hat.squeeze(0)  # Remove batch dim if exists
    print(f"[Debug]: y_tilda_hat shape: {y_tilda_hat.shape}")
    torch.cuda.empty_cache()

    # Convert numpy arrays to xarray DataArrays with proper coordinates
    # Get coordinates from the original dataset
    lat = ds.coords.get('lat', ds.coords.get('latitude'))
    lon = ds.coords.get('lon', ds.coords.get('longitude'))
    time_coords1 = ds.time.isel(time=(start_idx + seq_len - 1))
    time_coords2 = ds.time.isel(time=slice(start_idx, start_idx + seq_len))
    
    # Create DataArrays for each validation dataset
    y_da = xr.DataArray(
        y,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": time_coords1, # batch dimension but put time coordinates ^^
            "lat": lat,
            "lon": lon,
            "channel": np.arange(y.shape[1])
        },
        name="data"
    )
    
    y_hat_da = xr.DataArray(
        y_hat,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": time_coords2,
            "lat": lat,
            "lon": lon,
            "channel": np.arange(y_hat.shape[1])
        },
        name="data"
    )
    
    y_tilda_hat_da = xr.DataArray(
        y_tilda_hat,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": time_coords2,
            "lat": lat,
            "lon": lon,
            "channel": np.arange(y_tilda_hat.shape[1])
        },
        name="data"
    )
    
    # Concatenate along a new 'validation' dimension
    y_da = y_da.isel(time=0)
    print(f"[Debug]: y_da shape after isel: {y_da.shape}")
    y_hat_da = y_hat_da.isel(time=1)
    print(f"[Debug]: y_hat_da shape after isel: {y_hat_da.shape}")
    y_tilda_hat_da = y_tilda_hat_da.isel(time=1)
    print(f"[Debug]: y_tilda_hat_da shape after isel: {y_tilda_hat_da.shape}")
    val_ds = xr.concat([y_da, y_hat_da, y_tilda_hat_da], dim='validation')
    val_ds = val_ds.assign_coords(validation=['target', 'initial_forecast', 'optimized_forecast'])
    val_ds.to_netcdf(f"Validation_{state_number}_{date}_{windows}_days_window.nc")
    print(f"Saved: Validation dataset at ./Validation_{state_number}_{date}_{windows}_days_window.nc")
    
    return x_param.detach().clone().cpu().numpy()

if __name__ == "__main__":
    main()