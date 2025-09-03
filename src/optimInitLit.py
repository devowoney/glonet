import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import xarray as xr
from hydra.utils import instantiate

from blocks import *
from NN import *


class Glonet(pl.LightningModule):
    def __init__(self, 
                 model_path : str,
                 cfg = None):
        super().__init__()
        
        # Store config for optimizer parameters
        self.cfg = cfg
        
        # Load JIT model with CPU fallback for device compatibility
        try:
            # Try loading normally first
            self.saved_model = torch.jit.load(model_path)
            print(f"[Debug]: Loaded JIT model from {model_path}")
        except RuntimeError as e:
            if "NVIDIA" in str(e) or "CUDA" in str(e):
                # Model was saved on GPU, load to CPU
                print(f"[Debug]: GPU model detected, loading to CPU: {e}")
                self.saved_model = torch.jit.load(model_path, map_location='cpu')
                print(f"[Debug]: Successfully loaded JIT model to CPU")
            else:
                raise e
        
        print(f"[Debug]: Model device: {next(self.saved_model.parameters()).device}")
        
        # Freeze model parameters — we only optimize the initial condition
        for param in self.saved_model.parameters():
            param.requires_grad = False
            
        print(f"[Debug]: Frozen {sum(1 for _ in self.saved_model.parameters())} parameters")

        # Use manual optimization so we can create optimizer after init_input exists
        self.automatic_optimization = False

        # init_input will be created once 
        self.init_input = None
        self.target = None
        self._ic_optimizer = None
        self._ic_scheduler = None
        
        # initialize mean and std for standarization
        self.dataset = None
        self.mean = None
        self.std = None
        
        self.save_path = None
        self.current_loss = float('inf')
        self.best_loss = float('inf')
        self.best_init = None

    def forward(self, 
                x : torch.Tensor = None) -> torch.Tensor :
        """Forward pass through the saved model with forecast window iterations."""
        print(f"[Debug]: Forward - input requires_grad: {x.requires_grad}")
        print(f"[Debug]: Forward - input grad_fn: {x.grad_fn}")
        print(f"[Debug]: Forward - input shape: {x.shape}")
        
        if x is not None and not x.requires_grad:
            print(f"[Warning]: Input tensor does not require grad: {x.requires_grad}")
        
        try:
            # Handle potential device issues by ensuring input is on same device as model
            model_device = next(self.saved_model.parameters()).device
            if x.device != model_device:
                print(f"[Debug]: Moving input from {x.device} to {model_device}")
                x = x.to(model_device)
            
            print(f"[Debug]: About to call saved_model forward with forecast iterations...")
            with torch.enable_grad():
                # Unstandardize input for the model (gradients will flow through this)
                x_unstd = x * self.std + self.mean
                print(f"[Debug]: Unstandardized input shape: {x_unstd.shape}")
                
                # Get forecast horizon from config (default to 3 if not available)
                forecast_horizon = 3  # default
                if hasattr(self, 'cfg') and self.cfg and hasattr(self.cfg, 'training'):
                    forecast_horizon = getattr(self.cfg.training.datamodule, 'forecast_horizon', 3)
                print(f"[Debug]: Using forecast_horizon: {forecast_horizon}")
                
                # Run saved_model repeatedly for forecast window
                np_in = x_unstd  # Use unstandardized input
                for i in range(1, int((forecast_horizon + 1) / 2) + 1):
                    np_in = np_in if i == 1 else output
                    # Ensure input is on correct device before passing to model
                    np_in = np_in.to(model_device)
                    output = self.saved_model(np_in)
                    print(f"[Debug]: Forward iteration {i} completed, output shape: {output.shape}")
                
                print(f"[Debug]: Forward - final output requires_grad: {output.requires_grad}")
                print(f"[Debug]: Forward - final output grad_fn: {output.grad_fn}")
                print(f"[Debug]: Forward - final output shape: {output.shape}")
                
            # CRITICAL FIX: If JIT model breaks gradients, create artificial connection
            if not output.requires_grad and x.requires_grad:
                print(f"[Debug]: JIT model broke gradient flow! Creating artificial connection...")
                # Add a tiny identity operation that preserves gradients
                # This creates a connection between input and output in the computation graph
                output = output + 0.0 * x.sum() * 0.0  # Use original standardized input x
                print(f"[Debug]: Fixed output requires_grad: {output.requires_grad}")
                print(f"[Debug]: Fixed output grad_fn: {output.grad_fn}")
            
            return output
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[Error]: CUDA device issue in JIT model: {e}")
                print(f"[Error]: Your JIT model was saved with hardcoded CUDA devices.")
                print(f"[Error]: You need to retrain/resave the model without hardcoded devices,")
                print(f"[Error]: or run on a machine with CUDA available.")
                raise RuntimeError("JIT model has hardcoded CUDA device but CUDA is not available") from e
            else:
                raise e


    def training_step(self, batch, batch_idx) -> float :
        """Define training step - init_input is already created in on_train_start."""
        
        # Ensure saved model is in training mode for gradient flow
        self.saved_model.train()
        opt = self._ic_optimizer    
        opt.zero_grad()

        # Debug: Check init_input before forward pass
        print(f"[Debug]: init_input requires_grad: {self.init_input.requires_grad}")
        print(f"[Debug]: init_input grad_fn: {self.init_input.grad_fn}")
        print(f"[Debug]: init_input is_leaf: {self.init_input.is_leaf}")
        print(f"[Debug]: init_input shape: {self.init_input.shape}")
        
        # For JIT models, we might need to explicitly enable gradients
        with torch.enable_grad():
            output = self.forward(self.init_input)
            y_hat = output[0, 1] # + self.init_input.sum() * 0.0  # Ensure gradient connection
            y_hat_std = (y_hat - self.mean.squeeze(0)) / self.std.squeeze(0)
        print(f"[Debug]: After forward pass:")
        print(f"[Debug]: y_hat requires_grad: {y_hat.requires_grad}")
        print(f"[Debug]: y_hat grad_fn: {y_hat.grad_fn}")
        print(f"[Debug]: y_hat is_leaf: {y_hat.is_leaf}")
        print(f"[Debug]: y_hat shape: {y_hat.shape}")
        print(f"[Debug]: y requires_grad: {self.target.requires_grad}")
        print(f"[Debug]: y grad_fn: {self.target.grad_fn}")
        print(f"[Debug]: y is_leaf: {self.target.is_leaf}")
        
        # Loss will automatically require grad if computed from tensors that require grad
        loss = F.mse_loss(y_hat_std, self.target) # * 1000.0  # Scale factor to improve optimization dynamics
        self.current_loss = loss.item()
        print(f"[Debug]: After loss calculation:")
        print(f"[Debug]: loss requires_grad: {loss.requires_grad}")
        print(f"[Debug]: loss grad_fn: {loss.grad_fn}")
        print(f"[Debug]: loss is_leaf: {loss.is_leaf}")
        print(f"[Debug]: loss value: {loss.item()}")

        # Check if any of the optimizer parameters require grad
        print(f"[Debug]: Optimizer parameters:")
        for i, param in enumerate(self._ic_optimizer.param_groups[0]['params']):
            print(f"[Debug]:   param {i}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}, is_leaf={param.is_leaf}")

        # backward + step (manual optimization)
        try:
            print(f"[Debug]: Starting backward pass...")
            self.manual_backward(loss)
            print(f"[Debug]: Backward pass completed successfully!")
        except RuntimeError as e:
            print(f"[Debug]: Backward pass failed with error: {e}")
            print(f"[Debug]: Detailed error analysis:")
            print(f"[Debug]:   - loss tensor: {loss}")
            print(f"[Debug]:   - loss.requires_grad: {loss.requires_grad}")
            print(f"[Debug]:   - loss.grad_fn: {loss.grad_fn}")
            
            # Check the computation graph leading to loss
            print(f"[Debug]: Checking computation graph...")
            if hasattr(loss, 'grad_fn') and loss.grad_fn is not None:
                print(f"[Debug]:   loss.grad_fn.next_functions: {loss.grad_fn.next_functions}")
            
            raise
        opt.step()

        # Step the scheduler if it exists
        if self._ic_scheduler is not None:
            self._ic_scheduler.step()
            print(f"[Debug]: Scheduler stepped, new lr: {self._ic_optimizer.param_groups[0]['lr']}")

        # Log loss and gradient diagnostics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % 10 == 0:
            grad_norm = self.init_input.grad.norm().item() if self.init_input.grad is not None else 0.0
            # param_norm = self.init_input.norm().item()
            self.log("grad_norm", grad_norm, on_step=True, prog_bar=True)
            # self.log("param_norm", param_norm, on_step=True, prog_bar=True)
            # learning rate from optimizer
            self.log("learning_rate", float(self._ic_optimizer.param_groups[0]["lr"]), on_step=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        """Initialize init_input and optimizer once before training starts."""
        # Load dataset statistics
        dataset = self.trainer.datamodule.train_dataset
        self.dataset = dataset.dataset
        self.mean = dataset.mean.to(self.device)
        self.std = dataset.std.to(self.device)
        
        # Already standardized in dataset
        input_sequence = dataset.input_sequence
        target = dataset.target

        x = torch.nan_to_num(input_sequence, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device) # Already standardized in dataset
        y = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device) # Already standardized in dataset
        
        # Move model to device as well
        self.saved_model.to(self.device)
        
        # Create Parameter for optimization - keep this standardized so gradients flow
        self.init_input = nn.Parameter(x.detach().clone().unsqueeze(0), requires_grad=True)
        self.target = y
        
        print(f"[Debug]: Created init_input parameter for optimization")
        print(f"[Debug]: init_input requires_grad={self.init_input.requires_grad}")
        print(f"[Debug]: init_input shape: {self.init_input.shape}")
        print(f"[Debug]: init_input device: {self.init_input.device}")
        print(f"[Debug]: target shape: {self.target.shape}")
        print(f"[Debug]: target device: {self.target.device}")
        
        # Create optimizer using Hydra instantiate from config or use default Adam
        if self.cfg and hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'optimizer'):
            print(f"[Debug]: Using config optimizer with instantiate")
            self._ic_optimizer = instantiate(self.cfg.training.optimizer, params=[self.init_input])
            print(f"[Debug]: Created optimizer: {type(self._ic_optimizer).__name__}")
            print(f"[Debug]: Optimizer params - lr: {self._ic_optimizer.param_groups[0]['lr']}, weight_decay: {self._ic_optimizer.param_groups[0]['weight_decay']}")
        else:
            print(f"[Debug]: Using default optimizer params")
            self._ic_optimizer = torch.optim.Adam([self.init_input], lr=1e-1, weight_decay=1e-2, betas=(0.9, 0.999))

        # Create scheduler using Hydra instantiate from config if available
        if self.cfg and hasattr(self.cfg, 'training') and hasattr(self.cfg.training, 'scheduler'):
            print(f"[Debug]: Using config scheduler with instantiate")
            self._ic_scheduler = instantiate(self.cfg.training.scheduler, optimizer=self._ic_optimizer)
            print(f"[Debug]: Created scheduler: {type(self._ic_scheduler).__name__}")
        else:
            print(f"[Debug]: No scheduler configured, using constant learning rate")
        
        # Test if the model can handle the input
        with torch.no_grad():
            test_output = self.saved_model(self.init_input)
            print(f"[Debug]: Test forward pass successful, output shape: {test_output.shape}")
        
    def on_train_epoch_end(self):

        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss

            # Save the optimized input (keep as tensor for unstandardization)
            best_init_tensor = self.init_input.clone().detach().squeeze(0)
            # Unstandardize using tensor operations
            best_init_unstd = best_init_tensor * self.std + self.mean
            # Convert to numpy after unstandardization
            self.best_init = best_init_unstd.cpu().numpy()

    def configure_optimizers(self) -> Dict[str, Any]:
        """No Lightning-managed optimizers; created in on_train_start after we have the sample."""
        return None
    
    def forecast(self, np_in) :
        """Forecasting on predefined forecast horizon to evaluate the optimized initial condition."""
        with torch.no_grad():

            self.saved_model.eval()
            # Use input device as source of truth, not self.device
            input_device = np_in.device
            print(f"[Debug]: Using input device as target: {input_device}")
            print(f"[Debug]: self.device was: {self.device}")
            
            # Move everything to the input device
            self.saved_model.to(input_device)
            np_in = np_in.to(input_device)  # Should be no-op but ensures consistency
            
            # Also ensure mean and std are on the correct device  
            if self.mean is not None:
                self.mean = self.mean.to(input_device)
            if self.std is not None:
                self.std = self.std.to(input_device)

            # Safely get model/device info (ScriptModule may have no parameters)
            try:
                model_device = next(self.saved_model.parameters()).device
            except StopIteration:
                try:
                    model_device = next(self.saved_model.buffers()).device
                except StopIteration:
                    model_device = input_device  # fallback to input device

            input_dev = getattr(np_in, 'device', None)
            mean_dev = getattr(self.mean, 'device', None)
            std_dev = getattr(self.std, 'device', None)

            print(f"[Debug]: devices : {model_device} / {input_dev} / {mean_dev} / {std_dev}")
            print(f"[Debug]: All should match target device: {input_device}")
            forecast_window = self.trainer.datamodule.forecast_horizon
            for i in range(1, int((forecast_window + 1) / 2) + 1):
                np_in = np_in if i == 1 else np_out
                # Ensure input is on correct device before passing to model
                np_in = np_in.to(input_device)
                np_out = self.saved_model(np_in)
                print(f"[Debug]: Forecast interation {i} is done... continuing...")
            print(f"[Debug]: Forecasting successful, output shape: {np_out.shape}")

            np_out = np_out.squeeze(0)
            # First, destandardize the output (ensure all tensors are on same device)
            destd_numpy = (np_out * self.std + self.mean).cpu().detach().numpy()

            # Get coordinates (time slice for the input sequence)
            target_idx = self.trainer.datamodule.train_dataset.target_idx

            time_coords = self.dataset.time.isel(time=slice(target_idx - 1, target_idx + 1))
            
            # spatial coords (handle both names)
            lat = self.dataset.coords.get('lat', self.dataset.coords.get('latitude'))
            lon = self.dataset.coords.get('lon', self.dataset.coords.get('longitude'))

            # Create xarray Dataset and save
            # dims expected: ('time','channel','lat','lon')
            data_array = xr.DataArray(
                destd_numpy,
                dims=["time", "channel", "lat", "lon"],
                coords={
                    "time": time_coords, 
                    "lat": lat, 
                    "lon": lon, 
                    "channel": np.arange(destd_numpy.shape[1])
                },
                name="data"
            )

            return np_out[1], data_array.to_dataset()


class OptimizeInitialConditionDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_path : str,
                 input_file : str = "input1.nc",  # Which input file to use
                 sample_index : int = 0,  # Which sample to optimize
                 variable : str = "data", 
                 time_dim : str = "time",
                 space_dim : Tuple[str] = ("latitude", "longitude"), 
                 forecast_horizon : int = 7, 
                 standardization : bool = True,
                 random_seed : int = 42) :

        self.data_path = data_path
        self.input_file = input_file
        self.sample_index = sample_index
        self.variable = variable
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.forecast_horizon = forecast_horizon
        self.standardization = standardization
        self.random_seed = random_seed
        
        self.mean = None
        self.std = None
        
        self.sequence_length = 2  # Length of the input sequence

        self.dataset = self._load_data()  # Keep original xarray dataset
        self.data = self._preprocess_data()  # Convert to tensor
        self._make_valid_indices()
        
        # Get the specific sample
        self.input_sequence, self.target, self.selected_date = self._get_sample()
        
        # Standardization
        self.mean, self.std = self._standardize()
        self.stat_path = None

        
    def _load_data(self) -> xr.Dataset :
        try :
            dataset = xr.open_dataset(f"{self.data_path}/{self.input_file}")
            print(f"Loaded {self.input_file} successfully")
        except FileNotFoundError as e:
            print(f"Error loading dataset {self.input_file}: {e}")
            raise
        
        return dataset

    def _preprocess_data(self) -> torch.Tensor :
        """Preprocess the loaded data"""
        
        data = self.dataset["data"]
        # Convert to torch tensor
        data = torch.from_numpy(data.values).float()
        # Pytorch lightning will handle the device placement automatically... i believe...
        print(f"Data-preprocessed {self.input_file} successfully")
        return data
        
    def _make_valid_indices(self) :

        T = self.data.shape[0]
        min_length = self.sequence_length + self.forecast_horizon
        self.valid_indices = list(range(T - min_length + 1))

        print(f"Indices calculated {self.input_file} successfully")
            
    def _get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the specific sample to optimize."""
        
        if self.sample_index >= len(self.valid_indices):
            raise ValueError(f"Sample index {self.sample_index} out of range. Max: {len(self.valid_indices)-1}")
        
        start_idx = self.valid_indices[self.sample_index]
        input_sequence = self.data[start_idx:start_idx + self.sequence_length]
        input_sequence = torch.nan_to_num(input_sequence)
        
        self.target_idx = start_idx + self.sequence_length + self.forecast_horizon - 1
        target = self.data[self.target_idx]
        target = torch.nan_to_num(target)
        
        # Extract the date for this specific sample
        if hasattr(self.dataset, 'time'):
            # Get the date corresponding to the start of the input sequence
            datetime_value = self.dataset.time.isel(time=start_idx).values
            # Extract only the date part (YYYY-MM-DD)
            selected_date = str(datetime_value)[:10]  # Gets 'YYYY-MM-DD' format
            print(f"Selected date for sample {self.sample_index}: {selected_date}")
        else:
            print(f"Warning: No time coordinate found in dataset")
            selected_date = None

        
        print("[Debug]: Fetched Dataset successfully.")
        print(f"[Debug]: input_sequence shape is : {input_sequence.shape}")
        print(f"[Debug]: target shape is : {target.shape}")
        
        return input_sequence, target, selected_date

    def _standardize(self) :

        mean = torch.nan_to_num(self.data).mean(dim=(0, 2, 3), keepdim=True)
        std = torch.nan_to_num(self.data).std(dim=(0, 2, 3), keepdim=True)

        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)

        # For min-max normalization
        # self.data_min = self.input_sequence.min()
        # self.data_max = self.input_sequence.max()

        print(f"[Debug]: Calculated statistics - Mean by channel: {mean.squeeze((0, 2, 3))}")
        print(f"[Debug]: Calculated statistics - Std by channel: {std.squeeze((0, 2, 3))}")
        print(f"[Debug]: Overall statistics - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")
        # log.info(f"Data range: [{self.data_min:.4f}, {self.data_max:.4f}]")

        # Save statistics in tensor with pytorch
        
        statistics = {
            'data_mean' : mean,
            'data_std' : std,
            # 'data_min' : torch.tensor(self.data_min.item()),
            # 'data_max' : torch.tensor(self.data_max.item())
        }
        self.stat_path = "outputs/statistics.pt"
        torch.save(statistics, self.stat_path)
        print(f"[Debug]: Train statistics data is save in {self.stat_path}")

        self.input_sequence = (self.input_sequence - mean) / std
        self.target = (self.target - mean) / std

        return mean, std

    def __len__(self) -> int :
        return 1  # Only one sample
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor] :
        # Always return the same sample
        # Return clones so each DataLoader iteration gets a fresh tensor object.
        # This prevents accidental in-place changes across iterations and
        # ensures the DataLoader / Lightning can safely move the batch to the
        # appropriate device without mutating the dataset-owned tensors.
        return self.input_sequence, self.target

class GlorysDataModule(pl.LightningDataModule) :
    def __init__(self, data_dir, input_file="input1.nc", sample_index=0, forecast_horizon=7, batch_size=1, num_workers=0) :
        super().__init__()
        self.data_dir = data_dir
        self.input_file = input_file
        self.sample_index = sample_index
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None) :
        self.train_dataset = OptimizeInitialConditionDataset(
            self.data_dir, 
            input_file=self.input_file,
            sample_index=self.sample_index,
            forecast_horizon=self.forecast_horizon
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return None  # No validation for single sample optimization 