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
now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
DEFAULT_OUTPUT_DIR = f"/Odyssey/private/j25lee/glonet/training/outputs/man_optimIC_GD/{now}_uo"

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
                 device: str = "cuda:0"):
        
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
        self.loss_fn = nn.MSELoss()
        
        # Initialize variables
        self.x0_1 = None
        self.x0_2 = None
        self.x0_3 = None
        self.target1 = None
        self.target2 = None
        self.target3 = None
        self.ocean_mask_1 = None
        self.ocean_mask_2 = None
        self.ocean_mask_3 = None
        self.coords = None
        
        # Track best predictions
        self.best_loss = float('inf')
        self.best_y_hat1 = None
        self.best_y_hat2 = None
        self.best_y_hat3 = None
        
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
        
        # Split into three parts for three models
        input1 = input_data[:, 0:5, :, :]      # Surface
        input2 = input_data[:, 5:45, :, :]     # Shallow
        input3 = input_data[:, 45:85, :, :]    # Deep
        
        target1 = target_data[0:5, :, :]
        target2 = target_data[5:45, :, :]
        target3 = target_data[45:85, :, :]
        
        # Convert to torch tensors and add batch dimension
        self.x0_1 = torch.from_numpy(input1).float().unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        self.x0_2 = torch.from_numpy(input2).float().unsqueeze(0).to(self.device)
        self.x0_3 = torch.from_numpy(input3).float().unsqueeze(0).to(self.device)
        
        self.target1 = torch.from_numpy(target1).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
        self.target2 = torch.from_numpy(target2).float().unsqueeze(0).to(self.device)
        self.target3 = torch.from_numpy(target3).float().unsqueeze(0).to(self.device)
        
        
        # Make initial conditions require gradients
        self.x0_1.requires_grad = False
        self.x0_2.requires_grad = False
        self.x0_3.requires_grad = False
        
        self.x0_1_u = self.x0_1[:, :, 3:4, :, :].clone().detach()
        self.x0_2_u = self.x0_2[:, :, 20:30, :, :].clone().detach()
        self.x0_3_u = self.x0_3[:, :, 20:30, :, :].clone().detach()
        self.x0_1_u.requires_grad = True
        self.x0_2_u.requires_grad = True
        self.x0_3_u.requires_grad = True

        self.x0_1_left_1 = self.x0_1[:, :, 0:3, :, :].clone().detach()
        self.x0_1_left_2 = self.x0_1[:, :, 4:5, :, :].clone().detach()
        self.x0_2_left_1 = self.x0_2[:, :, 0:20, :, :].clone().detach()
        self.x0_2_left_2 = self.x0_2[:, :, 30:40, :, :].clone().detach()
        self.x0_3_left_1 = self.x0_3[:, :, 0:20, :, :].clone().detach()
        self.x0_3_left_2 = self.x0_3[:, :, 30:40, :, :].clone().detach()
        self.x0_1_left_1.requires_grad = False
        self.x0_1_left_2.requires_grad = False
        self.x0_2_left_1.requires_grad = False
        self.x0_2_left_2.requires_grad = False
        self.x0_3_left_1.requires_grad = False
        self.x0_3_left_2.requires_grad = False
        
        self.target_1_u = self.target1[:, 3:4, :, :]
        self.target_2_u = self.target2[:, 20:30, :, :]
        self.target_3_u = self.target3[:, 20:30, :, :]
        
        self.ocean_mask_1_u = self.ocean_mask_1[3:4, :, :]  # Only SSH channel
        self.ocean_mask_2_u = self.ocean_mask_2[20:30, :, :]
        self.ocean_mask_3_u = self.ocean_mask_3[20:30, :, :]
        
        log.info(f"Loaded data shapes:")
        log.info(f"  uo (x_1): {self.x0_1.shape}, Target for uo: {self.target_1_u.shape}")
        log.info(f"  uo (x_2): {self.x0_2.shape}, Target for uo: {self.target_2_u.shape}")
        log.info(f"  uo (x_3): {self.x0_3.shape}, Target for uo: {self.target_3_u.shape}")
        log.info(f"  Gradient tracking only for uo")
        # log.info(f"  Input 2: {self.x0_2.shape}, Target 2: {self.target2.shape}")
        # log.info(f"  Input 3: {self.x0_3.shape}, Target 3: {self.target3.shape}")
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the models."""

        with torch.enable_grad():
            # Normalize inputs (not targets - they stay in original space for loss calculation)
            # self.x0_1 = self.normalizer1(self.x0_1)
            self.x0_1_combined = torch.cat([self.x0_1_left_1, self.x0_1_u, self.x0_1_left_2], dim=2) 
            self.x1 = self.normalizer1(self.x0_1_combined)
            self.x0_2_combined = torch.cat([self.x0_2_left_1, self.x0_2_u, self.x0_2_left_2], dim=2)
            self.x2 = self.normalizer2(self.x0_2_combined)
            self.x0_3_combined = torch.cat([self.x0_3_left_1, self.x0_3_u, self.x0_3_left_2], dim=2)
            self.x3 = self.normalizer3(self.x0_3_combined)
            
        # Forward pass
        with torch.enable_grad():

            # y_hat1 = self.model1(self.x0_1)  # [1, T, C, H, W]
            y_hat1 = self.model1(self.x1) 
            y_hat2 = self.model2(self.x2)
            y_hat3 = self.model3(self.x3)
        
            # Extract the last timestep as prediction
            y_hat1 = y_hat1[:, -1, :, :, :]  # [1, C, H, W]
            y_hat2 = y_hat2[:, -1, :, :, :]
            y_hat3 = y_hat3[:, -1, :, :, :]
            
            # Denormalize predictions to match target space
            y_hat1 = self.denormalizer1(y_hat1)
            y_hat2 = self.denormalizer2(y_hat2)
            y_hat3 = self.denormalizer3(y_hat3)
            
            y_hat1 = y_hat1[:, 3:4, :, :]  # Retrive only thetao channel
            y_hat2 = y_hat2[:, 20:30, :, :]
            y_hat3 = y_hat3[:, 20:30, :, :]
            
        return y_hat1, y_hat2, y_hat3
    
    def compute_loss(self, y_hat1: torch.Tensor, y_hat2: torch.Tensor, y_hat3: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predictions and targets."""
        
        loss1 = self.loss_fn(y_hat1, self.target_1_u)
        loss2 = self.loss_fn(y_hat2, self.target_2_u)
        loss3 = self.loss_fn(y_hat3, self.target_3_u)
        
        total_loss = loss1 + loss2 + loss3
        return total_loss, loss1, loss2, loss3

    def compute_variable_errors(self, y_hat1: torch.Tensor, y_hat2: torch.Tensor, y_hat3: torch.Tensor) -> Dict:
        """Compute MSE and normalized MSE for each variable."""
        
        with torch.no_grad():
            # Concatenate all predictions and targets
            y_hat_all = torch.cat([y_hat1, y_hat2, y_hat3], dim=1)  # [1, 21, H, W]
            y_all = torch.cat([self.target_1_u, self.target_2_u, self.target_3_u], dim=1) # [1, 21, H, W]
            
            # Compute squared errors: (y - y_hat)^2
            squared_errors = (y_all - y_hat_all) ** 2 # [1, 21, H, W]
            
            # Compute variance of true values for normalization
            y_variance = torch.var(y_all, dim=(2, 3), keepdim=True) # [1, 21, H, W]
            
            # Apply ocean masks (combined for all parts)
            ocean_mask_all = torch.cat([self.ocean_mask_1_u, self.ocean_mask_2_u, self.ocean_mask_3_u], dim=0).unsqueeze(0)      
            # [1, 21, H, W]
            
            # Mask out land region
            squared_errors_masked = squared_errors * ocean_mask_all
            
            # Count valid ocean points per channel
            n_ocean_points = ocean_mask_all.sum(dim=(2, 3))  # [1, 21]
            
            # Compute mean squared error per channel
            mse_per_channel = squared_errors_masked.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 21]
            
            # compute variance of y over the same ocean points and same dims (spatial)
            # use population variance (unbiased=False) to match MSE's 1/N scaling
            # compute masked mean then masked var to ignore land
            masked_y = y_all * ocean_mask_all
            mean_y = masked_y.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 21]

            # compute variance: E[(y - mean)^2] over ocean points
            # expand mean to spatial dims for subtraction
            mean_y_exp = mean_y.unsqueeze(-1).unsqueeze(-1)  # [1, 21, 1, 1]
            sq_dev = ((y_all - mean_y_exp) ** 2) * ocean_mask_all
            var_per_channel = sq_dev.sum(dim=(2, 3)) / (n_ocean_points + 1e-10)  # [1, 21]

            # normalized MSE
            normalized_mse_per_channel = mse_per_channel / (var_per_channel + 1e-10)  # [1, 21]

            # Convert to numpy for easier indexing
            mse_np = mse_per_channel.cpu().numpy().squeeze(0)  # [21]
            norm_mse_np = normalized_mse_per_channel.cpu().numpy().squeeze(0)  # [21]
            
            # Extract errors for specific variables
            errors = {
                'uo': {
                    'channels': {
                        'surface': [0],
                        'shallow': list(range(1, 10)), 
                        'deep': list(range(10, 20)),   
                    },
                    'mse': {
                        'surface': float(mse_np[0]),
                        'shallow': mse_np[1:10].tolist(),
                        'deep': mse_np[10:20].tolist(),
                    },
                    'normalized_mse': {
                        'surface': float(norm_mse_np[0]),
                        'shallow': norm_mse_np[1:10].tolist(),
                        'deep': norm_mse_np[10:20].tolist(),
                    },
                    'mean_mse': {
                        'surface': float(mse_np[0]),
                        'shallow': float(mse_np[1:10].mean()),
                        'deep': float(mse_np[10:20].mean()),
                        'all': float(np.concatenate([mse_np[0:1], mse_np[1:10], mse_np[10:20]]).mean()),
                    },
                    'mean_normalized_mse': {
                        'surface': float(norm_mse_np[0]),
                        'shallow': float(norm_mse_np[1:10].mean()),
                        'deep': float(norm_mse_np[10:20].mean()),
                        'all': float(np.concatenate([norm_mse_np[0:1], norm_mse_np[1:10], norm_mse_np[10:20]]).mean()),
                    },
                    'rmse': {
                        'all' : float(np.sqrt(mse_np).mean())
                    }
                },
                
                # Store full channel-wise errors for reference
                'all_channels_mse': mse_np.tolist(),
                'all_channels_normalized_mse': norm_mse_np.tolist(),
            }
            
            return errors

    def print_error_analysis(self, errors: Dict) -> None:
        """Print formatted error analysis."""
        log.info("\n" + "=" * 80)
        log.info("ERROR ANALYSIS BY VARIABLE")
        log.info("=" * 80)
        
        # Current (uo)
        log.info("\n[uo - Current (Zonal Velocity)]")
        log.info(f"  Surface (ch=1):")
        log.info(f"    MSE:            {errors['uo']['mean_mse']['surface']:.6e}")
        log.info(f"    Normalized MSE: {errors['uo']['mean_normalized_mse']['surface']:.6f}")
        log.info(f"  Shallow (ch=2:11, 10 levels):")
        log.info(f"    Mean MSE:            {errors['uo']['mean_mse']['shallow']:.6e}")
        log.info(f"    Mean Normalized MSE: {errors['uo']['mean_normalized_mse']['shallow']:.6f}")
        log.info(f"  Deep (ch=11:20, 10 levels):")
        log.info(f"    Mean MSE:            {errors['uo']['mean_mse']['deep']:.6e}")
        log.info(f"    Mean Normalized MSE: {errors['uo']['mean_normalized_mse']['deep']:.6f}")
        log.info(f"  Overall Mean:")
        log.info(f"    Mean MSE:            {errors['uo']['mean_mse']['all']:.6e}")
        log.info(f"    Mean Normalized MSE: {errors['uo']['mean_normalized_mse']['all']:.6f} \n")


        
    def optimize(self) -> Dict:
        """Run manual gradient descent optimization."""
        log.info(f"\nStarting manual gradient descent optimization...")
        log.info(f"[!!] Note: Only salinity variable in Input 1 is being optimized.")
        log.info(f"Learning rate: {self.learning_rate}")
        log.info(f"Number of iterations: {self.num_iterations}")
        log.info("=" * 60)
        
        # Compute initial error analysis
        log.info("\nComputing initial error analysis...")
        y_hat1_init, y_hat2_init, y_hat3_init = self.forward()
        initial_errors = self.compute_variable_errors(y_hat1_init, y_hat2_init, y_hat3_init)
        self.print_error_analysis(initial_errors)

        loss_history = []
        
        for iteration in range(self.num_iterations):
            # Zero gradients manually
            if self.x0_1.grad is not None:
                self.x0_1.grad.zero_()
            if self.x0_2.grad is not None:
                self.x0_2.grad.zero_()
            if self.x0_3.grad is not None:
                self.x0_3.grad.zero_()
                
            if self.x0_1_u.grad is not None:
                self.x0_1_u.grad.zero_()
            if self.x0_2_u.grad is not None:
                self.x0_2_u.grad.zero_()
            if self.x0_3_u.grad is not None:
                self.x0_3_u.grad.zero_()    
            
            # Forward pass
            y_hat1, y_hat2, y_hat3 = self.forward()
            
            # Compute loss
            total_loss, loss1, loss2, loss3 = self.compute_loss(y_hat1, y_hat2, y_hat3)
            
            # Backward pass
            total_loss.backward()
            
            # Apply ocean mask to gradients (keep land at zero)
            with torch.no_grad():
                if self.x0_1.grad is not None:
                    self.x0_1.grad *= self.ocean_mask_1.unsqueeze(0).unsqueeze(0)
                if self.x0_2.grad is not None:
                    self.x0_2.grad *= self.ocean_mask_2.unsqueeze(0).unsqueeze(0)
                if self.x0_3.grad is not None:
                    self.x0_3.grad *= self.ocean_mask_3.unsqueeze(0).unsqueeze(0)
                    
                if self.x0_1_u.grad is not None:
                    self.x0_1_u.grad *= self.ocean_mask_1_u.unsqueeze(0).unsqueeze(0)
                if self.x0_2_u.grad is not None:
                    self.x0_2_u.grad *= self.ocean_mask_2_u.unsqueeze(0).unsqueeze(0)
                if self.x0_3_u.grad is not None:
                    self.x0_3_u.grad *= self.ocean_mask_3_u.unsqueeze(0).unsqueeze(0)
                               
            # Manual gradient descent update
            with torch.no_grad():
                if self.x0_1.grad is not None:
                    self.x0_1 -= self.learning_rate * self.x0_1.grad
                if self.x0_2.grad is not None:
                    self.x0_2 -= self.learning_rate * self.x0_2.grad
                if self.x0_3.grad is not None:
                    self.x0_3 -= self.learning_rate * self.x0_3.grad
                    
                if self.x0_1_u.grad is not None:
                    self.x0_1_u -= self.learning_rate * self.x0_1_u.grad
                if self.x0_2_u.grad is not None:
                    self.x0_2_u -= self.learning_rate * self.x0_2_u.grad
                if self.x0_3_u.grad is not None:
                    self.x0_3_u -= self.learning_rate * self.x0_3_u.grad
                    
            # Store loss
            loss_history.append(total_loss.item())
            
            # Update best predictions if this is the best loss so far
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                with torch.no_grad():
                    self.best_y_hat1 = y_hat1.detach().clone()
                    self.best_y_hat2 = y_hat2.detach().clone()
                    self.best_y_hat3 = y_hat3.detach().clone()
            
            # Compute gradient norms
            grad_norm_1 = self.x0_1.grad.norm().item() if self.x0_1.grad is not None else 0.0
            grad_norm_2 = self.x0_2.grad.norm().item() if self.x0_2.grad is not None else 0.0
            grad_norm_3 = self.x0_3.grad.norm().item() if self.x0_3.grad is not None else 0.0
            grad_norm_u_1 = self.x0_1_u.grad.norm().item() if self.x0_1_u.grad is not None else 0.0
            grad_norm_u_2 = self.x0_2_u.grad.norm().item() if self.x0_2_u.grad is not None else 0.0
            grad_norm_u_3 = self.x0_3_u.grad.norm().item() if self.x0_3_u.grad is not None else 0.0
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                log.info(f"Iteration {iteration + 1}/{self.num_iterations}")
                log.info(f"  Total Loss: {total_loss.item():.6f}")
                # log.info(f"  Grad Norms - Input1: {grad_norm_1:.6f}, Input2: {grad_norm_2:.6f}, Input3: {grad_norm_3:.6f}")
                log.info(f"  Grad Norm - x0u1: {grad_norm_u_1:.6f}, Grad Norm - x0u2: {grad_norm_u_2:.6f},"
                         f"Grad Norm - x0u3: {grad_norm_u_3:.6f}")
                log.info("-" * 60)
            
            # Memory cleanup
            del y_hat1, total_loss
            gc.collect()
            torch.cuda.empty_cache()
        
        log.info("=" * 60)
        log.info("Optimization completed!")
        log.info(f"Final loss: {loss_history[-1]:.6f}")
        log.info(f"Initial loss: {loss_history[0]:.6f}")
        log.info(f"Loss reduction: {(loss_history[0] - loss_history[-1]) / loss_history[0] * 100:.2f}%")
        
        # Compute final error analysis
        log.info("\nComputing final error analysis...")
        y_hat1_final, y_hat2_final, y_hat3_final = self.forward()
        final_errors = self.compute_variable_errors(y_hat1_final, y_hat2_final, y_hat3_final)
        self.print_error_analysis(final_errors)
        
        return {
            'loss_history': loss_history,
            'initial_errors': initial_errors,
            'final_errors': final_errors
        }
    
    def save_optimized_ic(self, output_dir: str) -> None:
        """Save optimized initial conditions as netCDF files."""
        log.info(f"\nSaving optimized initial conditions to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Denormalize the optimized initial conditions
        with torch.no_grad():
            opt_x0_1 = self.x0_1_combined.clone().detach()
            opt_x0_2 = self.x0_2_combined.clone().detach()
            opt_x0_3 = self.x0_3_combined.clone().detach()  
        
        # Convert to numpy and remove batch dimension
        # Apply ocean mask to set land points to NaN
        opt_x0_1_np = opt_x0_1.cpu().numpy().squeeze(0)  # [T, C, H, W]
        opt_x0_1_np[:, self.ocean_mask_1.cpu().numpy() == 0] = np.nan
        opt_x0_2_np = opt_x0_2.cpu().numpy().squeeze(0)
        opt_x0_2_np[:, self.ocean_mask_2.cpu().numpy() == 0] = np.nan
        opt_x0_3_np = opt_x0_3.cpu().numpy().squeeze(0)
        opt_x0_3_np[:, self.ocean_mask_3.cpu().numpy() == 0] = np.nan
        
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
    
    def save_best_predictions(self, output_dir: str) -> None:
        """Save best predictions (y_hat) as netCDF files."""
        log.info(f"\nSaving best predictions to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.best_y_hat1 is None:
            log.warning("No best predictions found to save.")
            return
        
        # Convert to numpy and remove batch dimension
        y_hat1_np = self.best_y_hat1.cpu().numpy().squeeze(0)  # [C, H, W]
        y_hat2_np = self.best_y_hat2.cpu().numpy().squeeze(0)
        y_hat3_np = self.best_y_hat3.cpu().numpy().squeeze(0)
        
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
    
    def save_error_analysis(self, results: Dict, output_dir: str) -> None:
        """Save error analysis to JSON file."""
        import json
        
        log.info(f"\nSaving error analysis to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
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
            'loss_history': results['loss_history'],
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
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index to use from dataset (default: 0)"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2,
        help="Length of input sequence (default: 2)"
    )
    
    parser.add_argument(
        "--forecast-horizon",
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
    log.info("=" * 60)
    
    # Initialize optimizer
    optimizer = ManualGradientDescent(
        data_path=args.data_path,
        model_location=args.model_location,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        sample_idx=args.sample_idx,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        device=args.device
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
