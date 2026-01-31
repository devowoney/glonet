"""
Manual Gradient Descent for Initial Condition Optimization
This script optimizes initial conditions (x_0) using manual gradient descent
with MSE loss function and fixed learning rate.
"""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Tuple, Dict
import gc
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add path to glonet modules
sys.path.append(str(Path(__file__).parent.parent / "glonet_daily_forecast_local"))
sys.path.append(str(Path(__file__).parent.parent / "src/glonet"))
sys.path.append(str(Path(__file__).parent / "src"))

from utility import get_normalizer1, get_normalizer2, get_normalizer3
from utility import get_denormalizer1, get_denormalizer2, get_denormalizer3
from modelp2 import Glonet
from optimIC_GD_glonetLit import GlonetGradientCheckpointing

# Constants
MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
now = datetime.now().strftime('%Y-%m-%d-%H%M%S_x2y')
DEFAULT_OUTPUT_DIR = f"/Odyssey/private/j25lee/glonet/gradient_descent_optimic/outputs/man_optimIC_GD/{now}"

# Setup logging
log = logging.getLogger(__name__)


class ManualGradientDescent:
    """Manual gradient descent optimizer for initial condition optimization."""
    
    def __init__(self, 
                 data_path: str,
                 model_location: str = MODEL_LOCATION,
                 learning_rate: float = 0.01,
                 num_iterations: int = 100,
                 patch_size: Tuple[int, int] = (96, 96),
                 enable_patching: bool = False,
                 sample_idx: int = 0,
                 sequence_length: int = 2,
                 forecast_horizon: int = 7,
                 device: str = "cuda:0",
                 tensorboard_dir: str = None):
        
        self.data_path = data_path
        self.model_location = model_location
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.patch_size = patch_size
        self.enable_patching = enable_patching
        self.sample_idx = sample_idx
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.tensorboard_dir = tensorboard_dir
        
        # Initialize TensorBoard writer
        self.writer = None
        if self.tensorboard_dir:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            log.info(f"TensorBoard logging enabled: {self.tensorboard_dir}")
        
        # Initialize normalizers and denormalizers
        log.info("Loading normalizers and denormalizers...")
        self.normalizer1 = get_normalizer1(model_location)
        self.normalizer2 = get_normalizer2(model_location)
        self.normalizer3 = get_normalizer3(model_location)
        
        self.denormalizer1 = get_denormalizer1(model_location)
        self.denormalizer2 = get_denormalizer2(model_location)
        self.denormalizer3 = get_denormalizer3(model_location)
        
        # Load models with gradient checkpointing
        log.info("Loading pretrained models with gradient checkpointing...")
        self.model1 = self._load_checkpoint_model(
            f"{model_location}/glonet_part1.pth",
            shape_in=(2, 5, 672, 1440) if not enable_patching else (2, 5, patch_size[0], patch_size[1])
        )
        self.model2 = self._load_checkpoint_model(
            f"{model_location}/glonet_part2.pth",
            shape_in=(2, 40, 672, 1440) if not enable_patching else (2, 40, patch_size[0], patch_size[1])
        )
        self.model3 = self._load_checkpoint_model(
            f"{model_location}/glonet_part3.pth",
            shape_in=(2, 40, 672, 1440) if not enable_patching else (2, 40, patch_size[0], patch_size[1])
        )
        
        # Freeze model parameters
        for model in [self.model1, self.model2, self.model3]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Loss function
        # self.loss_fn = nn.MSELoss()
        
        # Initialize variables
        # self.x0_1 = None
        # self.x0_2 = None
        # self.x0_3 = None
        # self.target1 = None
        # self.target2 = None
        # self.target3 = None
        self.x0 = None
        self.target = None
        self.ocean_mask_1 = None
        self.ocean_mask_2 = None
        self.ocean_mask_3 = None
        self.coords = None
        
        # Track best predictions
        self.optimal_perturb = None
        self.best_loss = float('inf')
        self.best_y_hat = None
        # self.best_y_hat1 = None
        # self.best_y_hat2 = None
        # self.best_y_hat3 = None
        self.ic_correction = None
        self.pred_correction = None
        self.pred_mismatch = None
        
    def _load_checkpoint_model(self, checkpoint_path: str, shape_in: Tuple[int, int, int, int]) -> torch.nn.Module:
        """Load a checkpoint-based model with gradient checkpointing."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create GlonetGradientCheckpointing model
        model = GlonetGradientCheckpointing(shape_in=shape_in)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to train mode for gradient checkpointing
        model = model.to(self.device)
        model.train()  # Use train mode to enable gradient checkpointing
        
        return model
    
    def _create_ocean_masks(self, data: xr.Dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create ocean masks based on NaN values in the data."""
        log.info("Creating ocean masks...")
        
        # Get a sample time slice
        sample_data = data.isel(time=self.sample_idx)
        data_values = sample_data['data'].values  # Shape: [C, H, W]
        
        # Create land masks (1 where NaN, 0 where valid)
        land_mask = np.isnan(data_values).astype(np.float32)
        
        # Create ocean masks (1 - land_mask)
        ocean_mask_1 = 1.0 - land_mask[0:5, :, :]      # Surface: channels 0-4
        ocean_mask_2 = 1.0 - land_mask[5:45, :, :]     # Shallow: channels 5-44
        ocean_mask_3 = 1.0 - land_mask[45:85, :, :]    # Deep: channels 45-84
        
        # Convert to torch tensors
        ocean_mask_1 = torch.from_numpy(ocean_mask_1.copy()).float().to(self.device)
        ocean_mask_2 = torch.from_numpy(ocean_mask_2.copy()).float().to(self.device)
        ocean_mask_3 = torch.from_numpy(ocean_mask_3.copy()).float().to(self.device)
        
        log.info(f"Ocean mask 1 shape: {ocean_mask_1.shape}")
        log.info(f"Ocean mask 2 shape: {ocean_mask_2.shape}")
        log.info(f"Ocean mask 3 shape: {ocean_mask_3.shape}")
        
        return ocean_mask_1, ocean_mask_2, ocean_mask_3
    
    def load_data(self) -> None:
        """Load dataset and prepare initial conditions and targets."""
        log.info(f"Loading dataset from {self.data_path}...")
        
        # Load dataset
        dataset = xr.open_dataset(self.data_path)
        
        # Create ocean masks
        self.ocean_mask_1, self.ocean_mask_2, self.ocean_mask_3 = self._create_ocean_masks(dataset)
        
        # Extract input sequence
        input_sequence = dataset.isel(
            time=slice(self.sample_idx, self.sample_idx + self.sequence_length)
        )
        
        # Extract target (forecast_horizon timesteps ahead)
        target_time_idx = self.sample_idx + self.sequence_length + self.forecast_horizon - 1
        target = dataset.isel(time=target_time_idx)
        
        # Store coordinates
        self.coords = {
            'time': input_sequence['time'].values,
            'target_time': target['time'].values,
            'lat': input_sequence['lat'].values,
            'lon': input_sequence['lon'].values
        }
        
        # Convert to numpy arrays
        input_data = input_sequence['data'].values  # Shape: [T, C, H, W]
        target_data = target['data'].values         # Shape: [C, H, W]
        
        # Replace NaN with 0
        input_data = np.nan_to_num(input_data, nan=0.0)
        target_data = np.nan_to_num(target_data, nan=0.0)
        
        # # Split into three parts for three models
        # input1 = input_data[:, 0:5, :, :]      # Surface
        # input2 = input_data[:, 5:45, :, :]     # Shallow
        # input3 = input_data[:, 45:85, :, :]    # Deep
        
        # target1 = target_data[0:5, :, :]
        # target2 = target_data[5:45, :, :]
        # target3 = target_data[45:85, :, :]
        
        # # Convert to torch tensors and add batch dimension
        # self.x0_1 = torch.from_numpy(input1).float().unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        # self.x0_2 = torch.from_numpy(input2).float().unsqueeze(0).to(self.device)
        # self.x0_3 = torch.from_numpy(input3).float().unsqueeze(0).to(self.device)
        
        # self.target1 = torch.from_numpy(target1).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
        # self.target2 = torch.from_numpy(target2).float().unsqueeze(0).to(self.device)
        # self.target3 = torch.from_numpy(target3).float().unsqueeze(0).to(self.device)
        
        # # Store reference initial conditions before normalization (for IC RMSE computation)
        # self.x0_1_ref = torch.from_numpy(input1).float().unsqueeze(0).to(self.device).clone()
        # self.x0_2_ref = torch.from_numpy(input2).float().unsqueeze(0).to(self.device).clone()
        # self.x0_3_ref = torch.from_numpy(input3).float().unsqueeze(0).to(self.device).clone()
        
        # # Normalize inputs (not targets - they stay in original space for loss calculation)
        # self.x0_1 = self.normalizer1(self.x0_1)
        # self.x0_2 = self.normalizer2(self.x0_2)
        # self.x0_3 = self.normalizer3(self.x0_3)
        
        # # Make initial conditions require gradients
        # self.x0_1.requires_grad = True
        # self.x0_2.requires_grad = True
        # self.x0_3.requires_grad = True
        
        self.x0 = torch.from_numpy(input_data.copy()).float().unsqueeze(0).to(self.device)  # [1, 2, 85, H, W]
        self.x0_ref = self.x0.clone()
        self.target = torch.from_numpy(target_data.copy()).float().unsqueeze(0).to(self.device)  # [1, 85, H, W]
        
        self.x0.requires_grad = True
        self.x0_ref.requires_grad = False
        
        log.info(f"Loaded data shapes:")
        # log.info(f"  Input 1: {self.x0_1.shape}, Target 1: {self.target1.shape}")
        # log.info(f"  Input 2: {self.x0_2.shape}, Target 2: {self.target2.shape}")
        # log.info(f"  Input 3: {self.x0_3.shape}, Target 3: {self.target3.shape}")
        log.info(f"  Input: {self.x0.shape}, Target: {self.target.shape}")
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the models."""
        # Apply ocean masks to inputs
        # x1 = self.x0_1 * self.ocean_mask_1.unsqueeze(0).unsqueeze(0)  # Broadcast to [1, T, C, H, W]
        # x2 = self.x0_2 * self.ocean_mask_2.unsqueeze(0).unsqueeze(0)
        # x3 = self.x0_3 * self.ocean_mask_3.unsqueeze(0).unsqueeze(0)
        
        with torch.enable_grad():
            x1 = x[:, :, 0:5, :, :] 
            x1_in_std = self.normalizer1(x1) * self.ocean_mask_1.unsqueeze(0).unsqueeze(0)
            x2 = x[:, :, 5:45, :, :]
            x2_in_std = self.normalizer2(x2)  * self.ocean_mask_2.unsqueeze(0).unsqueeze(0)
            x3 = x[:, :, 45:85, :, :]
            x3_in_std = self.normalizer3(x3)  * self.ocean_mask_3.unsqueeze(0).unsqueeze(0)

            # Forward pass
            y_hat1 = self.model1(x1_in_std)  # [1, T, C, H, W]
            y_hat2 = self.model2(x2_in_std)
            y_hat3 = self.model3(x3_in_std)
        
            # Extract the last timestep as prediction
            y_hat1 = y_hat1[:, -1, :, :, :]  # [1, C, H, W]
            y_hat2 = y_hat2[:, -1, :, :, :]
            y_hat3 = y_hat3[:, -1, :, :, :]
            
            # Denormalize predictions to match target space
            y_hat1 = self.denormalizer1(y_hat1)
            y_hat2 = self.denormalizer2(y_hat2)
            y_hat3 = self.denormalizer3(y_hat3)
        
        y_hat = torch.cat([y_hat1, y_hat2, y_hat3], dim=1)  # [1, 85, H, W]
        x0_in_std = torch.cat([x1_in_std, x2_in_std, x3_in_std], dim=2)
        
        return x0_in_std, y_hat
    
    def compute_loss(self, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        """Compute MSE loss between predictions and targets."""
        
        ocean_mask_all = torch.cat([self.ocean_mask_1,
                                    self.ocean_mask_2,
                                    self.ocean_mask_3], dim=0).unsqueeze(0)  # [1, 85, H, W]
        
        se = ((self.target - y_hat) ** 2)                             # [1, 85, 672, 1440]
        se_masked = se * ocean_mask_all                               # [1, 85, 672, 1440]
        n_ocean_points = ocean_mask_all.sum(dim=(2, 3))               # [1, 85]
        mse = se_masked.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)   # [1, 85]
        
        var = self.target.var(dim=(2, 3))     # [1, 85]
        nmse = mse / (var + 1e-10)            # [1, 85]
        
        return mse, nmse

    
    def compute_variable_errors(self, y_hat: torch.Tensor) -> Dict:
        """
        Compute per-variable error analysis.
        
        Variables:
        - SSH: ch=0 (in part1)
        - thetao (temperature): 
            * Surface (part1): ch=1 (1 level)
            * Shallow (part2): ch=5:14 (10 levels) 
            * Deep (part3): ch=45:54 (10 levels)
        - so (salinity):
            * Surface (part1): ch=2 (1 level)
            * Shallow (part2): ch=15:24 (10 levels)
            * Deep (part3): ch=55:64 (10 levels)
        - uo (eastward velocity):
            * Surface (part1): ch=3 (1 level)
            * Shallow (part2): ch=25:34 (10 levels)
            * Deep (part3): ch=65:74 (10 levels)
        - vo (northward velocity):
            * Surface (part1): ch=4 (1 level)
            * Shallow (part2): ch=35:44 (10 levels)
            * Deep (part3): ch=75:84 (10 levels)
        
        Returns:
            Dictionary with MSE and normalized MSE for each variable
        """
        with torch.no_grad():
            # Concatenate all predictions and targets
            # y_hat_all = torch.cat([y_hat1, y_hat2, y_hat3], dim=1)  # [1, 85, H, W]
            # y_all = torch.cat([self.target1, self.target2, self.target3], dim=1)  # [1, 85, H, W]
            
            # Compute squared errors: (y - y_hat)^2
            squared_errors = (self.target - y_hat) ** 2  # [1, 85, H, W]
            
            # Compute variance of true values for normalization
            y_variance = torch.var(self.target, dim=(2, 3), keepdim=True)  # [1, 85, 1, 1]
            
            # Apply ocean masks (combined for all parts)
            ocean_mask_all = torch.cat([
                self.ocean_mask_1,      # [5, H, W]
                self.ocean_mask_2,      # [40, H, W]
                self.ocean_mask_3       # [40, H, W]
            ], dim=0).unsqueeze(0)      # [1, 85, H, W]
            
            # Mask out land regions
            squared_errors_masked = squared_errors * ocean_mask_all
            
            # Count valid ocean points per channel
            n_ocean_points = ocean_mask_all.sum(dim=(2, 3))  # [1, 85]
            
            # Compute mean squared error per channel
            mse_per_channel = squared_errors_masked.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 85]
            
            # Compute RMSE per channel
            rmse_per_channel = torch.sqrt(mse_per_channel)  # [1, 85]
            
            # Compute variance manually for normalized RMSE
            masked_y = self.target * ocean_mask_all  # [1, 85, H, W]
            mean_y = masked_y.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 85]
            mean_y_exp = mean_y.unsqueeze(-1).unsqueeze(-1)  # [1, 85, 1, 1]
            variance_y = ((self.target - mean_y_exp) ** 2) * ocean_mask_all
            var_per_channel = variance_y.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 85]
            
            # Compute normalized RMSE (divide by standard deviation)
            normalized_rmse_per_channel = rmse_per_channel / (torch.sqrt(var_per_channel) + 1e-10)  # [1, 85]
            
            # Convert to numpy for easier indexing
            rmse_np = rmse_per_channel.cpu().numpy().squeeze(0)  # [85]
            norm_rmse_np = normalized_rmse_per_channel.cpu().numpy().squeeze(0)  # [85]
            
            # # Compute RMSE for initial conditions as reference
            # # Get current denormalized initial conditions
            # x0_1_current = self.denormalizer1(self.x0_1.detach())[:, -1, :, :, :]  # [1, 5, H, W] - last timestep
            # x0_2_current = self.denormalizer2(self.x0_2.detach())[:, -1, :, :, :]  # [1, 40, H, W]
            # x0_3_current = self.denormalizer3(self.x0_3.detach())[:, -1, :, :, :]  # [1, 40, H, W]
            # x0_current_all = torch.cat([x0_1_current, x0_2_current, x0_3_current], dim=1)  # [1, 85, H, W]
            
            # # Reference initial conditions (last timestep)
            # x0_ref_all = torch.cat([
            #     self.x0_1_ref[:, -1, :, :, :],
            #     self.x0_2_ref[:, -1, :, :, :],
            #     self.x0_3_ref[:, -1, :, :, :]
            # ], dim=1)  # [1, 85, H, W]
            
            # Compute IC squared errors
            ic_squared_errors = (self.x0_ref - self.x0) ** 2  # [1, 2, 85, H, W]
            ic_squared_errors_masked = ic_squared_errors * ocean_mask_all
            ic_mse_per_channel = ic_squared_errors_masked.sum(dim=(3, 4)) / (n_ocean_points + 1e-10)  # [1, 2, 85]
            ic_rmse_per_channel = torch.sqrt(ic_mse_per_channel)  # [1, 2, 85]
            
            # Compute variance of reference IC for normalization
            masked_x0_ref = self.x0_ref * ocean_mask_all
            mean_x0_ref = masked_x0_ref.sum(dim=(3, 4)) / (n_ocean_points + 1e-10)
            mean_x0_ref_exp = mean_x0_ref.unsqueeze(-1).unsqueeze(-1)
            variance_x0_ref = ((self.x0_ref - mean_x0_ref_exp) ** 2) * ocean_mask_all
            var_x0_ref_per_channel = variance_x0_ref.sum(dim=(3, 4)) / (n_ocean_points + 1e-10)
            
            ic_normalized_rmse_per_channel = ic_rmse_per_channel / (torch.sqrt(var_x0_ref_per_channel) + 1e-10)
            
            ic_rmse_np = ic_rmse_per_channel.cpu().numpy().squeeze(0)  # [2, 85]
            ic_normalized_rmse_np = ic_normalized_rmse_per_channel.cpu().numpy().squeeze(0)  # [2, 85]
            
            # Extract errors for specific variables
            errors = {
                'Prediction': {
                    'SSH': {
                        'channels': [0],
                        'rmse': float(rmse_np[0]),
                        'normalized_rmse': float(norm_rmse_np[0]),
                    },
                    'thetao': {
                        'channels': {
                            'surface': [1],
                            'shallow': list(range(5, 15)),
                            'deep': list(range(45, 55)),
                        },
                        'mean_rmse': {
                            'surface': float(rmse_np[1]),
                            'shallow': float(rmse_np[5:15].mean()),
                            'deep': float(rmse_np[45:55].mean()),
                            'all': float(np.concatenate([rmse_np[1:2], rmse_np[5:15], rmse_np[45:55]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(norm_rmse_np[1]),
                            'shallow': float(norm_rmse_np[5:15].mean()),
                            'deep': float(norm_rmse_np[45:55].mean()),
                            'all': float(np.concatenate([norm_rmse_np[1:2], norm_rmse_np[5:15], norm_rmse_np[45:55]]).mean()),
                        }
                    },
                    'so': {
                        'channels': {
                            'surface': [2],
                            'shallow': list(range(15, 25)),
                            'deep': list(range(55, 65)),
                        },
                        'mean_rmse': {
                            'surface': float(rmse_np[2]),
                            'shallow': float(rmse_np[15:25].mean()),
                            'deep': float(rmse_np[55:65].mean()),
                            'all': float(np.concatenate([rmse_np[2:3], rmse_np[15:25], rmse_np[55:65]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(norm_rmse_np[2]),
                            'shallow': float(norm_rmse_np[15:25].mean()),
                            'deep': float(norm_rmse_np[55:65].mean()),
                            'all': float(np.concatenate([norm_rmse_np[2:3], norm_rmse_np[15:25], norm_rmse_np[55:65]]).mean()),
                        }
                    },
                    'uo': {
                        'channels': {
                            'surface': [3],
                            'shallow': list(range(25, 35)),
                            'deep': list(range(65, 75)),
                        },
                        'mean_rmse': {
                            'surface': float(rmse_np[3]),
                            'shallow': float(rmse_np[25:35].mean()),
                            'deep': float(rmse_np[65:75].mean()),
                            'all': float(np.concatenate([rmse_np[3:4], rmse_np[25:35], rmse_np[65:75]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(norm_rmse_np[3]),
                            'shallow': float(norm_rmse_np[25:35].mean()),
                            'deep': float(norm_rmse_np[65:75].mean()),
                            'all': float(np.concatenate([norm_rmse_np[3:4], norm_rmse_np[25:35], norm_rmse_np[65:75]]).mean()),
                        }
                    },
                    'vo': {
                        'channels': {
                            'surface': [4],
                            'shallow': list(range(35, 45)),
                            'deep': list(range(75, 85)),
                        },
                        'mean_rmse': {
                            'surface': float(rmse_np[4]),
                            'shallow': float(rmse_np[35:45].mean()),
                            'deep': float(rmse_np[75:85].mean()),
                            'all': float(np.concatenate([rmse_np[4:5], rmse_np[35:45], rmse_np[75:85]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(norm_rmse_np[4]),
                            'shallow': float(norm_rmse_np[35:45].mean()),
                            'deep': float(norm_rmse_np[75:85].mean()),
                            'all': float(np.concatenate([norm_rmse_np[4:5], norm_rmse_np[35:45], norm_rmse_np[75:85]]).mean()),
                        }
                    },
                },
                'Initial Conditions': {
                    'SSH': {
                        'rmse': float(ic_rmse_np[1, 0]),
                        'normalized_rmse': float(ic_normalized_rmse_np[1, 0]),
                    },
                    'thetao': {
                        'mean_rmse': {
                            'surface': float(ic_rmse_np[1, 1]),
                            'shallow': float(ic_rmse_np[1, 5:15].mean()),
                            'deep': float(ic_rmse_np[1, 45:55].mean()),
                            'all': float(np.concatenate([ic_rmse_np[1, 1:2], ic_rmse_np[1, 5:15], ic_rmse_np[1, 45:55]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(ic_normalized_rmse_np[1, 1]),
                            'shallow': float(ic_normalized_rmse_np[1, 5:15].mean()),
                            'deep': float(ic_normalized_rmse_np[1, 45:55].mean()),
                            'all': float(np.concatenate([ic_normalized_rmse_np[1, 1:2], 
                                                         ic_normalized_rmse_np[1, 5:15], 
                                                         ic_normalized_rmse_np[1, 45:55]]).mean()),
                        }
                    },
                    'so': {
                        'mean_rmse': {
                            'surface': float(ic_rmse_np[1, 2]),
                            'shallow': float(ic_rmse_np[1, 15:25].mean()),
                            'deep': float(ic_rmse_np[1, 55:65].mean()),
                            'all': float(np.concatenate([ic_rmse_np[1, 2:3], ic_rmse_np[1, 15:25], ic_rmse_np[1, 55:65]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(ic_normalized_rmse_np[1, 2]),
                            'shallow': float(ic_normalized_rmse_np[1, 15:25].mean()),
                            'deep': float(ic_normalized_rmse_np[1, 55:65].mean()),
                            'all': float(np.concatenate([ic_normalized_rmse_np[1, 2:3], 
                                                         ic_normalized_rmse_np[1, 15:25], 
                                                         ic_normalized_rmse_np[1, 55:65]]).mean()),
                        }
                    },
                    'uo': {
                        'mean_rmse': {
                            'surface': float(ic_rmse_np[1, 3]),
                            'shallow': float(ic_rmse_np[1, 25:35].mean()),
                            'deep': float(ic_rmse_np[1, 65:75].mean()),
                            'all': float(np.concatenate([ic_rmse_np[1, 3:4], ic_rmse_np[1, 25:35], ic_rmse_np[1, 65:75]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(ic_normalized_rmse_np[1, 3]),
                            'shallow': float(ic_normalized_rmse_np[1, 25:35].mean()),
                            'deep': float(ic_normalized_rmse_np[1, 65:75].mean()),
                            'all': float(np.concatenate([ic_normalized_rmse_np[1, 3:4], 
                                                         ic_normalized_rmse_np[1, 25:35], 
                                                         ic_normalized_rmse_np[1, 65:75]]).mean()),
                        }
                    },
                    'vo': {
                        'mean_rmse': {
                            'surface': float(ic_rmse_np[1, 4]),
                            'shallow': float(ic_rmse_np[1, 35:45].mean()),
                            'deep': float(ic_rmse_np[1, 75:85].mean()),
                            'all': float(np.concatenate([ic_rmse_np[1, 4:5], ic_rmse_np[1, 35:45], ic_rmse_np[1, 75:85]]).mean()),
                        },
                        'mean_normalized_rmse': {
                            'surface': float(ic_normalized_rmse_np[1, 4]),
                            'shallow': float(ic_normalized_rmse_np[1, 35:45].mean()),
                            'deep': float(ic_normalized_rmse_np[1, 75:85].mean()),
                            'all': float(np.concatenate([ic_normalized_rmse_np[1, 4:5], 
                                                         ic_normalized_rmse_np[1, 35:45], 
                                                         ic_normalized_rmse_np[1, 75:85]]).mean()),
                        }
                    },
                },
                # Store full channel-wise errors for reference
                'all_channels_rmse': rmse_np.tolist(),
                'all_channels_normalized_rmse': norm_rmse_np.tolist(),
                'ic_all_channels_rmse': ic_rmse_np.tolist(),
                'ic_all_channels_normalized_rmse': ic_normalized_rmse_np.tolist(),
            }
            
            return errors
    
    def print_error_analysis(self, errors: Dict) -> None:
        """Print formatted error analysis."""
        log.info("\n" + "=" * 80)
        log.info("ERROR ANALYSIS BY VARIABLE")
        log.info("=" * 80)
        
        # SSH
        log.info("\n[SSH - Sea Surface Height] (ch=0)")
        log.info(
            f"  Prediction RMSE:            "
            f"{errors['Prediction']['SSH']['rmse']:.6e}"
        )
        log.info(
            f"  Prediction Normalized RMSE: "
            f"{errors['Prediction']['SSH']['normalized_rmse']:.6f}"
        )
        log.info(
            f"  IC RMSE:                    "
            f"{errors['Initial Conditions']['SSH']['rmse']:.6e}"
        )
        log.info(
            f"  IC Normalized RMSE:         "
            f"{errors['Initial Conditions']['SSH']['normalized_rmse']:.6f}"
        )
        
        # Temperature (thetao)
        log.info("\n[THETAO - Temperature]")
        log.info(
            f"  Prediction - Overall Mean RMSE:            "
            f"{errors['Prediction']['thetao']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  Prediction - Overall Mean Normalized RMSE: "
            f"{errors['Prediction']['thetao']['mean_normalized_rmse']['all']:.6f}"
        )
        log.info(
            f"  IC - Overall Mean RMSE:                    "
            f"{errors['Initial Conditions']['thetao']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  IC - Overall Mean Normalized RMSE:         "
            f"{errors['Initial Conditions']['thetao']['mean_normalized_rmse']['all']:.6f}"
        )
        
        # Salinity (so)
        log.info("\n[SO - Salinity]")
        log.info(
            f"  Prediction - Overall Mean RMSE:            "
            f"{errors['Prediction']['so']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  Prediction - Overall Mean Normalized RMSE: "
            f"{errors['Prediction']['so']['mean_normalized_rmse']['all']:.6f}"
        )
        log.info(
            f"  IC - Overall Mean RMSE:                    "
            f"{errors['Initial Conditions']['so']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  IC - Overall Mean Normalized RMSE:         "
            f"{errors['Initial Conditions']['so']['mean_normalized_rmse']['all']:.6f}"
        )
        
        # Eastward velocity (uo)
        log.info("\n[UO - Eastward Velocity]")
        log.info(
            f"  Prediction - Overall Mean RMSE:            "
            f"{errors['Prediction']['uo']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  Prediction - Overall Mean Normalized RMSE: "
            f"{errors['Prediction']['uo']['mean_normalized_rmse']['all']:.6f}"
        )
        log.info(
            f"  IC - Overall Mean RMSE:                    "
            f"{errors['Initial Conditions']['uo']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  IC - Overall Mean Normalized RMSE:         "
            f"{errors['Initial Conditions']['uo']['mean_normalized_rmse']['all']:.6f}"
        )
        
        # Northward velocity (vo)
        log.info("\n[VO - Northward Velocity]")
        log.info(
            f"  Prediction - Overall Mean RMSE:            "
            f"{errors['Prediction']['vo']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  Prediction - Overall Mean Normalized RMSE: "
            f"{errors['Prediction']['vo']['mean_normalized_rmse']['all']:.6f}"
        )
        log.info(
            f"  IC - Overall Mean RMSE:                    "
            f"{errors['Initial Conditions']['vo']['mean_rmse']['all']:.6e}"
        )
        log.info(
            f"  IC - Overall Mean Normalized RMSE:         "
            f"{errors['Initial Conditions']['vo']['mean_normalized_rmse']['all']:.6f}"
        )
        
        log.info("\n" + "=" * 80)
    
    def optimize(self) -> Dict:
        """Run manual gradient descent optimization."""
        log.info(f"\nStarting manual gradient descent optimization...")
        log.info(f"Learning rate: {self.learning_rate}")
        log.info(f"Number of iterations: {self.num_iterations}")
        log.info("=" * 60)
        
        # Compute initial error analysis
        log.info("\nComputing initial error analysis...")
        x0_in_std, y_hat_initial = self.forward(self.x0)
        initial_errors = self.compute_variable_errors(y_hat_initial)
        self.print_error_analysis(initial_errors)
        
        loss_history = []
        
        for iteration in range(self.num_iterations):
            # Zero gradients manually
            # if self.x0_1.grad is not None:
            #     self.x0_1.grad.zero_()
            # if self.x0_2.grad is not None:
            #     self.x0_2.grad.zero_()
            # if self.x0_3.grad is not None:
            #     self.x0_3.grad.zero_()
            if self.x0.grad is not None:
                self.x0.grad.zero_()
            
            # Forward pass
            x0_in_std, y_hat = self.forward(self.x0)
            
            # Compute loss
            total_loss, total_nloss = self.compute_loss(y_hat)
            
            nloss_input1 = total_nloss[:, 0:5]
            nloss_input2 = total_nloss[:, 5:45]
            nloss_input3 = total_nloss[:, 45:85]
            
            nloss_ssh = total_nloss[:, 0:1]
            nloss_t = torch.cat([total_nloss[:, 1:2], total_nloss[:, 5:15], total_nloss[:, 45:55]], dim=1)
            nloss_s = torch.cat([total_nloss[:, 2:3], total_nloss[:, 15:25], total_nloss[:, 55:65]], dim=1)
            nloss_u = torch.cat([total_nloss[:, 3:4], total_nloss[:, 25:35], total_nloss[:, 65:75]], dim=1)
            nloss_v = torch.cat([total_nloss[:, 4:5], total_nloss[:, 35:45], total_nloss[:, 75:85]], dim=1)
            
            # Backward pass - only use input1 (surface) for gradient computation
            grads = torch.autograd.grad(nloss_input1.sum(), self.x0, create_graph=False)
            
            # Apply ocean mask to gradients and update
            with torch.no_grad():
                ocean_mask_all = torch.cat([self.ocean_mask_1,
                                            self.ocean_mask_2,
                                            self.ocean_mask_3], dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, 85, H, W]
                masked_grad = grads[0].detach().clone() * ocean_mask_all
                
                # Gradient descent update
                self.x0 = self.x0 - self.learning_rate * masked_grad
                self.x0.requires_grad = True
            
            # Clean up gradient computation immediately
            del grads, x0_in_std
            torch.cuda.empty_cache()
                
            # Store loss (detach to avoid keeping computation graph)
            loss_history.append(total_nloss.detach().clone())
            
            # Update best predictions if this is the best loss so far
            if total_nloss.sum().item() < self.best_loss:
                self.best_loss = total_nloss.sum().item()
                with torch.no_grad():
                    ocean_mask_all_noseq = torch.cat([self.ocean_mask_1,
                                                       self.ocean_mask_2,
                                                       self.ocean_mask_3], dim=0).unsqueeze(0)  # [1, 85, H, W]
                    
                    self.optimized_input = self.x0.detach().clone()
                    self.best_y_hat = y_hat.detach().clone()
                    
                    self.optimal_perturb = (self.optimized_input - self.x0_ref).detach() * ocean_mask_all
                    self.pred_correction = (self.best_y_hat - y_hat_initial).detach() * ocean_mask_all_noseq
                    self.pred_mismatch = (self.best_y_hat - self.target).detach() * ocean_mask_all_noseq
            
            # Compute error metrics for TensorBoard logging
            iteration_errors = self.compute_variable_errors(y_hat)
            
            # Log to TensorBoard
            if self.writer is not None:
                # Loss
                self.writer.add_scalar('Loss/total_loss', total_nloss.sum().item(), iteration)
                self.writer.add_scalar('Loss/loss_ssh', nloss_ssh.sum().item(), iteration)
                self.writer.add_scalar('Loss/loss_t', nloss_t.sum().item(), iteration)
                self.writer.add_scalar('Loss/loss_s', nloss_s.sum().item(), iteration)
                self.writer.add_scalar('Loss/loss_u', nloss_u.sum().item(), iteration)
                self.writer.add_scalar('Loss/loss_v', nloss_v.sum().item(), iteration)
                
                # Prediction RMSE by variable
                self.writer.add_scalar('RMSE/Prediction/SSH', 
                                       iteration_errors['Prediction']['SSH']['rmse'], iteration)
                self.writer.add_scalar('RMSE/Prediction/thetao', 
                                       iteration_errors['Prediction']['thetao']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/Prediction/so', 
                                       iteration_errors['Prediction']['so']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/Prediction/uo', 
                                       iteration_errors['Prediction']['uo']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/Prediction/vo', 
                                       iteration_errors['Prediction']['vo']['mean_rmse']['all'], iteration)
                
                # Initial Condition RMSE by variable
                self.writer.add_scalar('RMSE/InitialCondition/SSH', 
                                       iteration_errors['Initial Conditions']['SSH']['rmse'], iteration)
                self.writer.add_scalar('RMSE/InitialCondition/thetao', 
                                       iteration_errors['Initial Conditions']['thetao']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/InitialCondition/so', 
                                       iteration_errors['Initial Conditions']['so']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/InitialCondition/uo', 
                                       iteration_errors['Initial Conditions']['uo']['mean_rmse']['all'], iteration)
                self.writer.add_scalar('RMSE/InitialCondition/vo', 
                                       iteration_errors['Initial Conditions']['vo']['mean_rmse']['all'], iteration)
                
                # Normalized RMSE
                self.writer.add_scalar('RMSE_Normalized/Prediction/SSH', 
                                       iteration_errors['Prediction']['SSH']['normalized_rmse'], iteration)
                self.writer.add_scalar('RMSE_Normalized/InitialCondition/SSH', 
                                       iteration_errors['Initial Conditions']['SSH']['normalized_rmse'], iteration)
            
            # Compute gradient norms
            grad_norm = masked_grad.norm().item()
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                log.info(f"Iteration {iteration + 1}/{self.num_iterations}")
                log.info(f"  Total Loss: {total_nloss.sum().item():.6f}")
                log.info(f"  Loss ssh: {nloss_ssh.sum().item():.6f}, Loss thetao: {nloss_t.sum().item():.6f}")
                log.info(f"  Loss so: {nloss_s.sum().item():.6f}, "
                         f"Loss uo: {nloss_u.sum().item():.6f}, Loss vo: {nloss_v.sum().item():.6f}")
                log.info(f"  Prediction RMSE - SSH: {iteration_errors['Prediction']['SSH']['rmse']:.6e}")
                log.info(f"  IC RMSE - SSH: {iteration_errors['Initial Conditions']['SSH']['rmse']:.6e}")
                log.info(f"  Grad Norm: {grad_norm:.6f}")
                log.info("-" * 60)
            
            # Memory cleanup
            del y_hat, total_nloss, total_loss, masked_grad, ocean_mask_all
            del nloss_input1, nloss_input2, nloss_input3
            del nloss_ssh, nloss_t, nloss_s, nloss_u, nloss_v
            gc.collect()
            torch.cuda.empty_cache()
        
        log.info("="*60)
        log.info("Optimization completed!")
        log.info(f"Final loss: {loss_history[-1].sum():.6f}")
        log.info(f"Initial loss: {loss_history[0].sum():.6f}")
        log.info(f"Loss reduction: {(loss_history[0].sum() - loss_history[-1].sum()) / loss_history[0].sum() * 100:.2f}%")
        
        # Compute final error analysis
        log.info("\nComputing final error analysis...")
        x0_in_std, y_hat_final = self.forward(self.x0)
        final_errors = self.compute_variable_errors(y_hat_final)
        self.print_error_analysis(final_errors)
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            log.info("TensorBoard logs saved successfully")
        
        return {
            'loss_history': loss_history,
            'initial_errors': initial_errors,
            'final_errors': final_errors
        }
    
    def save_optimized_ic(self, output_dir: str) -> None:
        """Save optimized initial conditions as netCDF files."""
        log.info(f"\nSaving optimized initial conditions to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # # Denormalize the optimized initial conditions
        # with torch.no_grad():
        #     opt_x0_1 = self.denormalizer1(self.x0_1.detach())
        #     opt_x0_2 = self.denormalizer2(self.x0_2.detach())
        #     opt_x0_3 = self.denormalizer3(self.x0_3.detach())
        
        # Convert to numpy and remove batch dimension
        opt_x0_1_np = self.optimized_input[:, :, 0:5, :, :].clone().detach().cpu().numpy().squeeze(0)  # [T, C, H, W]
        opt_x0_2_np = self.optimized_input[:, :, 5:45, :, :].clone().detach().cpu().numpy().squeeze(0)
        opt_x0_3_np = self.optimized_input[:, :, 45:85, :, :].clone().detach().cpu().numpy().squeeze(0)
        
        # Get dimensions
        time, channel1, height, width = opt_x0_1_np.shape
        _, channel2, _, _ = opt_x0_2_np.shape
        
        # Create coordinate dictionaries
        coords1 = {
            'time': self.coords['time'],
            'ch': np.arange(channel1),
            'lat': self.coords['lat'],
            'lon': self.coords['lon']
        }
        coords2 = {
            'time': self.coords['time'],
            'ch': np.arange(channel2),
            'lat': self.coords['lat'],
            'lon': self.coords['lon']
        }
        
        # Create xarray Datasets
        ds1 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], opt_x0_1_np)
        }, coords=coords1)
        
        ds2 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], opt_x0_2_np)
        }, coords=coords2)
        
        ds3 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], opt_x0_3_np)
        }, coords=coords2)
        
        # Add metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, ds in enumerate([ds1, ds2, ds3], start=1):
            ds.attrs['description'] = f'Optimized initial condition part {i} from manual gradient descent'
            ds.attrs['creation_date'] = timestamp
            ds.attrs['learning_rate'] = self.learning_rate
            ds.attrs['num_iterations'] = self.num_iterations
            ds['data'].attrs['long_name'] = f'Optimized initial condition part {i}'
            ds['data'].attrs['units'] = 'model_units'
        
        # Save with compression
        encoding = {'data': {'zlib': True, 'complevel': 4}}
        
        path1 = f"{output_dir}/optimal_input1.nc"
        path2 = f"{output_dir}/optimal_input2.nc"
        path3 = f"{output_dir}/optimal_input3.nc"
        
        ds1.to_netcdf(path1, encoding=encoding)
        ds2.to_netcdf(path2, encoding=encoding)
        ds3.to_netcdf(path3, encoding=encoding)
        
        log.info(f"Saved optimized initial conditions:")
        log.info(f"  {path1} - shape {opt_x0_1_np.shape}")
        log.info(f"  {path2} - shape {opt_x0_2_np.shape}")
        log.info(f"  {path3} - shape {opt_x0_3_np.shape}")
        
        # Save IC correction fields as well
        ic_correction_np = self.optimal_perturb.cpu().numpy().squeeze(0)  # [T, C, H, W] where T=2
        
        cr1 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], ic_correction_np[:, 0:5, :, :])
        }, coords=coords1)
        cr2 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], ic_correction_np[:, 5:45, :, :])
        }, coords=coords2)
        cr3 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], ic_correction_np[:, 45:85, :, :])
        }, coords=coords2)
        
        cr1.to_netcdf(f"{output_dir}/ic_correction_1.nc", encoding=encoding)
        cr2.to_netcdf(f"{output_dir}/ic_correction_2.nc", encoding=encoding)
        cr3.to_netcdf(f"{output_dir}/ic_correction_3.nc", encoding=encoding)
        
        log.info("Saved initial condition correction fields.")
    
    def save_best_predictions(self, output_dir: str) -> None:
        """Save best predictions (y_hat) as netCDF files."""
        log.info(f"\nSaving best predictions to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.best_y_hat is None:
            log.warning("No best predictions found to save.")
            return
        
        # Convert to numpy and remove batch dimension
        y_hat1_np = self.best_y_hat[:, 0:5, :, :].cpu().numpy().squeeze(0)  # [C, H, W]
        y_hat2_np = self.best_y_hat[:, 5:45, :, :].cpu().numpy().squeeze(0)
        y_hat3_np = self.best_y_hat[:, 45:85, :, :].cpu().numpy().squeeze(0)
        
        # Add time dimension (expand to [1, C, H, W])
        y_hat1_np = np.expand_dims(y_hat1_np, axis=0)
        y_hat2_np = np.expand_dims(y_hat2_np, axis=0)
        y_hat3_np = np.expand_dims(y_hat3_np, axis=0)
        
        # Get dimensions
        time_dim, channel1, height, width = y_hat1_np.shape
        _, channel2, _, _ = y_hat2_np.shape
        _, channel3, _, _ = y_hat3_np.shape
        
        # Get target time
        target_time = np.array([self.coords['target_time']])
        
        # Create coordinate dictionaries
        coords1 = {
            'time': target_time,
            'ch': np.arange(channel1),
            'lat': self.coords['lat'],
            'lon': self.coords['lon']
        }
        coords2 = {
            'time': target_time,
            'ch': np.arange(channel2),
            'lat': self.coords['lat'],
            'lon': self.coords['lon']
        }
        coords3 = {
            'time': target_time,
            'ch': np.arange(channel3),
            'lat': self.coords['lat'],
            'lon': self.coords['lon']
        }
        
        # Create xarray Datasets
        ds1 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], y_hat1_np)
        }, coords=coords1)
        
        ds2 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], y_hat2_np)
        }, coords=coords2)
        
        ds3 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], y_hat3_np)
        }, coords=coords3)
        
        # Add metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, ds in enumerate([ds1, ds2, ds3], start=1):
            ds.attrs['description'] = f'Best forecast prediction part {i} from manual gradient descent optimization'
            ds.attrs['creation_date'] = timestamp
            ds.attrs['best_loss'] = float(self.best_loss)
            ds.attrs['learning_rate'] = self.learning_rate
            ds.attrs['num_iterations'] = self.num_iterations
            ds.attrs['sample_idx'] = self.sample_idx
            ds.attrs['sequence_length'] = self.sequence_length
            ds.attrs['forecast_horizon'] = self.forecast_horizon
            ds['data'].attrs['long_name'] = f'Best forecast prediction part {i}'
            ds['data'].attrs['units'] = 'model_units'
        
        # Save with compression
        encoding = {'data': {'zlib': True, 'complevel': 4}}
        
        path1 = f"{output_dir}/best_forecast_1.nc"
        path2 = f"{output_dir}/best_forecast_2.nc"
        path3 = f"{output_dir}/best_forecast_3.nc"
        
        ds1.to_netcdf(path1, encoding=encoding)
        ds2.to_netcdf(path2, encoding=encoding)
        ds3.to_netcdf(path3, encoding=encoding)
        
        log.info(f"Saved best predictions (loss={self.best_loss:.6f}):")
        log.info(f"  {path1} - shape {y_hat1_np.shape}")
        log.info(f"  {path2} - shape {y_hat2_np.shape}")
        log.info(f"  {path3} - shape {y_hat3_np.shape}")
        
        # Save correction fields as well
        pred_correction_np = self.pred_correction.cpu().numpy().squeeze(0)  # [C, H, W]
        pred_correction_np = np.expand_dims(pred_correction_np, axis=0)  # [1, C, H, W]
        
        ds_cr1 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_correction_np[:, 0:5, :, :])
        }, coords=coords1)
        
        ds_cr2 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_correction_np[:, 5:45, :, :])
        }, coords=coords2)
        
        ds_cr3 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_correction_np[:, 45:85, :, :])
        }, coords=coords3)
        
        path_cr1 = f"{output_dir}/prediction_correction_1.nc"
        path_cr2 = f"{output_dir}/prediction_correction_2.nc"
        path_cr3 = f"{output_dir}/prediction_correction_3.nc"
        
        ds_cr1.to_netcdf(path_cr1, encoding=encoding)
        ds_cr2.to_netcdf(path_cr2, encoding=encoding)
        ds_cr3.to_netcdf(path_cr3, encoding=encoding)
        
        log.info(f"Saved prediction correction fields.")
        
        # Same for prediction mismatch
        pred_mismatch_np = self.pred_mismatch.cpu().numpy().squeeze(0)  # [C, H, W]
        pred_mismatch_np = np.expand_dims(pred_mismatch_np, axis=0)  # [1, C, H, W]
        
        ds_mm1 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_mismatch_np[:, 0:5, :, :])
        }, coords=coords1)  
        ds_mm2 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_mismatch_np[:, 5:45, :, :])
        }, coords=coords2)
        ds_mm3 = xr.Dataset({
            'data': (['time', 'ch', 'lat', 'lon'], pred_mismatch_np[:, 45:85, :, :])
        }, coords=coords3)
        
        ds_mm1.to_netcdf(f"{output_dir}/prediction_mismatch_1.nc", encoding=encoding)
        ds_mm2.to_netcdf(f"{output_dir}/prediction_mismatch_2.nc", encoding=encoding)
        ds_mm3.to_netcdf(f"{output_dir}/prediction_mismatch_3.nc", encoding=encoding)
        
        log.info("Saved prediction mismatch fields.")
        
    def save_error_analysis(self, results: Dict, output_dir: str) -> None:
        """Save error analysis to JSON file."""
        import json
        
        log.info(f"\nSaving error analysis to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        loss_history_values = [float(loss.sum().item()) for loss in results['loss_history']]
        
        # Prepare data for JSON serialization
        error_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'configuration': {
                'learning_rate': self.learning_rate,
                'num_iterations': self.num_iterations,
                'sample_idx': self.sample_idx,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
            },
            'best_loss': float(self.best_loss),
            'initial_errors': results['initial_errors'],
            'final_errors': results['final_errors'],
            'loss_history': loss_history_values,
        }

        output_path = f"{output_dir}/error_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        log.info(f"Saved error analysis to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual Gradient Descent for Initial Condition Optimization"
    )
    
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to input netCDF dataset"
    )
    
    parser.add_argument(
        "-l", "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent (default: 0.01)"
    )
    
    parser.add_argument(
        "-n", "--num-iterations",
        type=int,
        default=100,
        help="Number of optimization iterations (default: 100)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for optimized initial conditions (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "-m", "--model-location",
        type=str,
        default=MODEL_LOCATION,
        help=f"Path to trained model weights (default: {MODEL_LOCATION})"
    )
    
    parser.add_argument(
        "-i", "--sample-idx",
        type=int,
        default=0,
        help="Sample index to use from dataset (default: 0)"
    )
    
    parser.add_argument(
        "-s", "--sequence-length",
        type=int,
        default=2,
        help="Length of input sequence (default: 2)"
    )
    
    parser.add_argument(
        "-f", "--forecast-horizon",
        type=int,
        default=7,
        help="Forecast horizon (default: 7)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/optimization.log")
        ]
    )
    
    log.info("=" * 60)
    log.info("Manual Gradient Descent for Initial Condition Optimization")
    log.info("=" * 60)
    log.info(f"Data path: {args.data_path}")
    log.info(f"Model location: {args.model_location}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Learning rate: {args.learning_rate}")
    log.info(f"Number of iterations: {args.num_iterations}")
    log.info(f"Device: {args.device}")
    log.info("="*60)
    
    # Setup TensorBoard directory
    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    
    # Initialize optimizer
    optimizer = ManualGradientDescent(
        data_path=args.data_path,
        model_location=args.model_location,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        sample_idx=args.sample_idx,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        device=args.device,
        tensorboard_dir=tensorboard_dir
    )
    
    # Load data
    optimizer.load_data()
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save error analysis
    optimizer.save_error_analysis(results, args.output_dir)
    
    # Save optimized initial conditions
    optimizer.save_optimized_ic(args.output_dir)
    
    # Save best predictions
    optimizer.save_best_predictions(args.output_dir)
    
    log.info("\n" + "=" * 60)
    log.info("Done!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
