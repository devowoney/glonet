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
    src_model = Glonet(dim=(2, 5, 672, 1440)).to(device)
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
    except RuntimeError as e:
        print(f"Error moving to device: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        src_model.to(device)
        input_sequence = input_sequence.unsqueeze(0).to(device)
        target = target.to(device)
        mean = mean.to(device)
        std = std.to(device)
    
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
    
    # Define optimizer
    optimizer = torch.optim.SGD([x_param], lr=1e7, momentum=0.7)
    
    # Reset gradients
    src_model.zero_grad()

    start_norm = x_param.data.norm().item()
    print(f'start x norm: {start_norm:.6f}')

    # small training loop
    steps = 100
    for i in range(steps):
        optimizer.zero_grad()
        out = src_model(x_param)
        
        loss = torch.nn.functional.mse_loss(out, target)  # normalized loss
        loss.backward()

        if i % 100 == 0 or i == steps - 1:
            grad_norm = x_param.grad.norm().item() if x_param.grad is not None else float('nan')
            print(f'step {i:02d} loss={loss.item():.6e} x.grad_norm={grad_norm:.6e} x.norm={x_param.data.norm().item():.6f}')

        optimizer.step()

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
    
    return x_param.detach().clone().cpu().numpy()

if __name__ == "__main__":
    main()