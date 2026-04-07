#!/usr/bin/env python
"""
GPU-Accelerated Ensemble Generation Script

This script uses:
- Dask arrays for lazy loading and memory efficiency
- PyTorch tensors for GPU-accelerated computation
- Data parallelism for batch processing multiple ensemble members simultaneously
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import dask.array as da

warnings.filterwarnings('ignore')


class GPUInitialConditionDataLoader :
    """
    GPU-accelerated data loader using Dask for lazy loading and PyTorch for computation.
    """
    
    def __init__(self, 
                 data_path: Union[str, List[str]], 
                 target_idx: int,
                 climato_path: str, 
                 rand_ds_path: str,
                 device: torch.device = None) -> None :
        
        self.data_path = data_path
        self.target_idx = target_idx
        self.climato_path = climato_path
        self.rand_ds_path = rand_ds_path
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load with Dask arrays (lazy loading)
        self.dataset = self.load_dataset_lazy(data_path)
        self.uv_depth_t0, self.uv_depth_t1 = self.get_ocean_current_lazy(self.dataset, target_idx)
        
    def load_dataset_lazy(self, 
                          data_path: Union[str, List[str]]) -> xr.Dataset :
        """Load dataset with Dask arrays for lazy evaluation"""
        chunks = {"time": 2, "lat": 96, "lon": 96, "ch": 10}
        
        if isinstance(data_path, list) :
            ds = xr.open_mfdataset(data_path, chunks=chunks, parallel=True)
            ds = ds.isel(time=slice(self.target_idx - 1, self.target_idx + 1))
        else:
            ds = xr.open_dataset(data_path, chunks=chunks)
            ds = ds.isel(time=slice(self.target_idx - 1, self.target_idx + 1))
        
        return ds
    
    def get_ocean_current_lazy(self, 
                               ds: xr.Dataset, 
                               target_idx: int) -> Tuple[dict, dict]:
        """
        Extract ocean current data keeping Dask arrays (no computation yet).
        """
        idx_t0 = target_idx - 1
        idx_t1 = target_idx
        
        # Extract u,v components (keep as Dask arrays)
        uv_t0 = ds["data"].isel(time=idx_t0, ch=slice(3, 5))
        uv_t1 = ds["data"].isel(time=idx_t1, ch=slice(3, 5))
        
        # Store statistics as Dask arrays (lazy)
        uv_depth_t0 = {
            "d0": uv_t0,
            "max_d0": uv_t0.max(dim=("lat", "lon"), skipna=True),
            "min_d0": uv_t0.min(dim=("lat", "lon"), skipna=True),
            "mean_d0": uv_t0.mean(dim=("lat", "lon"), skipna=True),
            "std_d0": uv_t0.std(dim=("lat", "lon"), skipna=True)
        }
        
        # Depth levels 1-20
        for depth in range(1, 11) :
            uv_depth_da = ds["data"].isel(time=idx_t0, ch=[25 + depth, 34 + depth])
            uv_depth_t0[f"d{depth}"] = uv_depth_da
            uv_depth_t0[f"max_d{depth}"] = uv_depth_da.max(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"min_d{depth}"] = uv_depth_da.min(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"mean_d{depth}"] = uv_depth_da.mean(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"std_d{depth}"] = uv_depth_da.std(dim=("lat", "lon"), skipna=True)
            
        for depth in range(11, 21) :
            uv_depth_da = ds["data"].isel(time=idx_t0, ch=[55 + depth, 64 + depth])
            uv_depth_t0[f"d{depth}"] = uv_depth_da
            uv_depth_t0[f"max_d{depth}"] = uv_depth_da.max(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"min_d{depth}"] = uv_depth_da.min(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"mean_d{depth}"] = uv_depth_da.mean(dim=("lat", "lon"), skipna=True)
            uv_depth_t0[f"std_d{depth}"] = uv_depth_da.std(dim=("lat", "lon"), skipna=True)
        
        # Same for t1
        uv_depth_t1 = {
            "d0": uv_t1,
            "max_d0": uv_t1.max(dim=("lat", "lon"), skipna=True),
            "min_d0": uv_t1.min(dim=("lat", "lon"), skipna=True),
            "mean_d0": uv_t1.mean(dim=("lat", "lon"), skipna=True),
            "std_d0": uv_t1.std(dim=("lat", "lon"), skipna=True)
        }
        
        for depth in range(1, 11) :
            uv_depth_da = ds["data"].isel(time=idx_t1, ch=[25 + depth, 34 + depth])
            uv_depth_t1[f"d{depth}"] = uv_depth_da
            uv_depth_t1[f"max_d{depth}"] = uv_depth_da.max(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"min_d{depth}"] = uv_depth_da.min(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"mean_d{depth}"] = uv_depth_da.mean(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"std_d{depth}"] = uv_depth_da.std(dim=("lat", "lon"), skipna=True)
            
        for depth in range(11, 21) :
            uv_depth_da = ds["data"].isel(time=idx_t1, ch=[55 + depth, 64 + depth])
            uv_depth_t1[f"d{depth}"] = uv_depth_da
            uv_depth_t1[f"max_d{depth}"] = uv_depth_da.max(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"min_d{depth}"] = uv_depth_da.min(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"mean_d{depth}"] = uv_depth_da.mean(dim=("lat", "lon"), skipna=True)
            uv_depth_t1[f"std_d{depth}"] = uv_depth_da.std(dim=("lat", "lon"), skipna=True)
        
        return uv_depth_t0, uv_depth_t1
    
    def get_ssh_anomaly(self) -> xr.DataArray :
        """Extract SSH anomaly (lazy)"""
        climato = xr.open_dataset(self.climato_path, chunks={'lat': 96, 'lon': 96})
        rand_ds = xr.open_dataset(self.rand_ds_path, chunks={'lat': 96, 'lon': 96})
        
        anomaly_ssh = climato.isel(time=0, ch=0)['data'] - rand_ds.isel(time=0, ch=0)['data']
        return anomaly_ssh


class GPUStochasticNoiseGenerator:
    """
    GPU-accelerated stochastic noise generation using PyTorch.
    """
    
    def __init__(self,
                 ssh_anomaly: xr.DataArray,
                 is_pointwise: bool,
                 seed: int = 42,
                 device: torch.device = None,
                 amplitude: float = 0.1) -> None :
        
        self.is_pointwise = is_pointwise
        self.seed = seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amplitude = amplitude
        
        # Convert to PyTorch tensor and move to GPU
        self.anomaly_ssh = torch.from_numpy(ssh_anomaly.values).float().to(self.device)
        self.nx = ssh_anomaly.sizes['lon']
        self.ny = ssh_anomaly.sizes['lat']
        
        # Generate noise on GPU
        self.ocean_regions = self.ocean_region_definiition()
        self.noise = self.create_noise_gpu(self.nx, self.ny, self.ocean_regions, self.seed)
    
    def ocean_region_definiition(self) :
        """Define ocean regions for noise generation"""
        ocean_regions = [
            {'name': 'Gulf Stream', 'type': 'box', 'correlation_length': 4.0,
             'coords': {'y_range': [382, 542], 'x_range': [300, 620]},
             'blend': True, 'blend_width': 70},
            {'name': 'Kuroshio', 'type': 'box', 'correlation_length': 4.0,
             'coords': {'y_range': [382, 542], 'x_range': [1160, 1460]},
             'blend': True, 'blend_width': 70},
            {'name': 'Agulhas', 'type': 'box', 'correlation_length': 4.0,
             'coords': {'y_range': [82, 242], 'x_range': [680, 980]},
             'blend': True, 'blend_width': 70},
            {'name': 'Antarctic ACC', 'type': 'box', 'correlation_length': 10.0,
             'coords': {'y_range': [80, 162], 'x_range': [0, self.nx]},
             'blend': True, 'blend_width': 30},
        ]
        return ocean_regions
    
    def create_noise_gpu(self, 
                         nx: int, 
                         ny: int, 
                         ocean_regions: dict, 
                         seed: int) -> torch.Tensor :
        """Create stochasticity on GPU"""
        torch.manual_seed(seed)
        
        if not self.is_pointwise :
            # Regional noise with GPU acceleration
            L_field_regions = self.create_correlation_length_field_gpu((ny, nx), 
                                                                       ocean_regions, 
                                                                       50)
            
            # Amplitude field based on SSH anomaly
            valid_mask = ~torch.isnan(self.anomaly_ssh)
            ssh_std = torch.std(self.anomaly_ssh[valid_mask]).item()
            amp_field_regions = torch.abs(self.anomaly_ssh) / ssh_std
            
            # Generate spatially-varying noise on GPU
            regional_noise = self.spatially_varying_gaussian_filter_gpu(
                (ny, nx), L_field_regions, amplitude=self.amplitude, seed=seed
            )
            
            regional_noise_scaled = regional_noise * amp_field_regions
            return regional_noise_scaled
        else:
            # Spectral noise
            k = torch.normal(mean=-2.0, std=0.3, size=(1,), 
                           generator=torch.Generator().manual_seed(4000 + seed)).item()
            noise_spectral = self.spectral_correlated_noise_gpu(
                (ny, nx), correlation_length=10, amplitude=self.amplitude, 
                spectral_slope=k, seed=4000 + seed
            )
            return noise_spectral
    
    def create_correlation_length_field_gpu(self, 
                                            shape, 
                                            regions, 
                                            default) :
        """Create correlation length field on GPU"""
        ny, nx = shape
        L_field = torch.ones((ny, nx), device=self.device) * default
        
        y, x = torch.meshgrid(
            torch.arange(ny, device=self.device),
            torch.arange(nx, device=self.device),
            indexing='ij'
        )
        
        for region in regions :
            mask = torch.zeros((ny, nx), device=self.device)
            reg_type = region['type']
            correlation_length = region.get('correlation_length', 20.0)
            blend = region.get('blend', True)
            blend_width = region.get('blend_width', 30)
            
            if reg_type == 'box' :
                coords = region['coords']
                y_min, y_max = coords['y_range']
                x_min, x_max = coords['x_range']
                
                box_mask = ((y >= y_min) & (y < y_max) & 
                           (x >= x_min) & (x < x_max)).float()
                
                if blend :
                    # Simple distance-based blending on GPU
                    mask = box_mask
                    # Simplified blending for GPU efficiency
                    from torch.nn.functional import avg_pool2d
                    kernel_size = min(blend_width, ny // 4)
                    if kernel_size > 1:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                        mask = avg_pool2d(mask, kernel_size, stride=1, 
                                        padding=kernel_size//2)
                        mask = mask.squeeze()
                        # Ensure mask has the correct shape after pooling
                        if mask.shape != (ny, nx):
                            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                size=(ny, nx), 
                                                mode='bilinear', 
                                                align_corners=True).squeeze()
                else:
                    mask = box_mask
            
            L_field = L_field * (1 - mask) + correlation_length * mask
        
        return L_field
    
    def spatially_varying_gaussian_filter_gpu(self, 
                                              shape : Tuple[int, int],
                                              correlation_length_field : torch.Tensor, 
                                              amplitude : float = 1.0, 
                                              seed: int = None) -> torch.Tensor :
        """GPU-accelerated spatially-varying Gaussian filter"""
        ny, nx = shape
        valid_mask = ~torch.isnan(correlation_length_field)
        L_min = torch.min(correlation_length_field[valid_mask]).item()
        L_max = torch.max(correlation_length_field[valid_mask]).item()
        
        n_scales = 5
        L_scales = torch.logspace(np.log10(max(L_min, 1)), np.log10(L_max), 
                                  n_scales, device=self.device)
        
        # Generate white noise on GPU
        if seed is not None:
            torch.manual_seed(seed)
        white_noise = torch.randn(ny, nx, device=self.device)
        
        # Apply Gaussian blur at different scales using PyTorch
        noise_scales = []
        for L in L_scales :
            sigma = L.item()
            # Use PyTorch's gaussian blur
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0 :
                kernel_size += 1
            kernel_size = min(kernel_size, min(ny, nx) // 2 * 2 - 1)
            
            if kernel_size >= 3 :
                filtered = self.gaussian_blur_2d_gpu(white_noise.unsqueeze(0).unsqueeze(0), 
                                                     kernel_size, sigma)
                noise_scales.append(filtered.squeeze())
            else:
                noise_scales.append(white_noise)
        
        # Blend scales
        output = torch.zeros((ny, nx), device=self.device)
        total_weight = torch.zeros((ny, nx), device=self.device)
        
        for i, L in enumerate(L_scales):
            log_L = torch.log(L)
            log_L_field = torch.log(torch.clamp(correlation_length_field, min=1e-6))
            weight = torch.exp(-((log_L_field - log_L)**2) / (2 * 0.5**2))
            
            output += weight * noise_scales[i]
            total_weight += weight
        
        output = output / (total_weight + 1e-10)
        output = torch.where(torch.isnan(correlation_length_field), 
                           torch.tensor(float('nan'), device=self.device), output)
        
        valid_mask = ~torch.isnan(output)
        valid_std = torch.std(output[valid_mask])
        output = output / valid_std * amplitude / 10
        
        return output
    
    def gaussian_blur_2d_gpu(self, 
                             tensor: torch.Tensor, 
                             kernel_size: int, 
                             sigma: float) -> torch.Tensor :
        """Apply Gaussian blur on GPU using PyTorch"""
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        
        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(tensor, gauss_2d, padding=padding)
        
        return blurred
    
    def spectral_correlated_noise_gpu(self, 
                                      shape: Tuple[int, int], 
                                      correlation_length: float, 
                                      amplitude: float = 1.0, 
                                      spectral_slope: float = -3, 
                                      seed: int = None) -> torch.Tensor :
        """GPU-accelerated spectral noise generation"""
        ny, nx = shape
        
        # Create frequency grid on GPU
        ky = torch.fft.fftfreq(ny, d=1.0, device=self.device)
        kx = torch.fft.fftfreq(nx, d=1.0, device=self.device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        K = torch.sqrt(KX**2 + KY**2)
        K[0, 0] = 1.0
        
        # Generate white noise
        if seed is not None:
            torch.manual_seed(seed)
        white_noise = torch.randn(ny, nx, device=self.device)
        white_spectrum = torch.fft.fft2(white_noise)
        
        # Apply spectral filter
        k0 = 1.0 / correlation_length
        spectral_filter = K**spectral_slope * torch.exp(-K / k0)
        spectral_filter[0, 0] = 0
        
        filtered_spectrum = white_spectrum * spectral_filter
        correlated_noise = torch.real(torch.fft.ifft2(filtered_spectrum))
        
        # Normalize
        correlated_noise = correlated_noise / torch.std(correlated_noise) * amplitude
        
        # Make positive and normalize
        correlated_noise_positive = torch.exp(correlated_noise)
        adjust_factor = (4 + spectral_slope) + 1e-6
        correlated_noise_positive_normalized = correlated_noise_positive / adjust_factor
        
        return correlated_noise_positive_normalized


class GPUCurrentBasedPerturbationGenerator :
    """
    GPU-accelerated perturbation generator with batch processing support.
    """
    
    def __init__(self, 
                 data_path: Union[str, List[str]], 
                 target_idx: int,
                 climato_path: str,
                 rand_ds_path: str,
                 dt: float,
                 device: torch.device = None,
                 batch_size: int = 4,
                 onenoise_amplitude: float = 0.1,
                 pointwisenoise_amplitude: float = 0.1) -> None :

        self.dt = dt
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.onenoise_amplitude = onenoise_amplitude
        self.pointwisenoise_amplitude = pointwisenoise_amplitude
        
        # Load data with lazy evaluation
        ic_dataloader = GPUInitialConditionDataLoader(data_path, 
                                                      target_idx, 
                                                      climato_path, 
                                                      rand_ds_path, 
                                                      self.device)
        self.dataset = ic_dataloader.dataset
        self.uv_depth_t0 = ic_dataloader.uv_depth_t0
        self.uv_depth_t1 = ic_dataloader.uv_depth_t1
        self.ssh_anomaly = ic_dataloader.get_ssh_anomaly()
    
    @staticmethod
    def create_land_mask_gpu(data_array_2d: xr.DataArray, 
                            threshold=1e-10, 
                            device=None) -> torch.Tensor:
        """Create land mask on GPU"""
        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(data_array_2d.dims) == 2:
            sample_data = torch.from_numpy(data_array_2d.values).float().to(device)
        else:
            raise ValueError("Input data_array must be 2D (lat, lon).")
        
        land_mask = torch.isnan(sample_data) | (torch.abs(sample_data) < threshold)
        return land_mask
    
    def displace_array_gpu(self, 
                          data: torch.Tensor,
                          dx: torch.Tensor,
                          dy: torch.Tensor,
                          land_mask: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated displacement using PyTorch grid_sample.
        Supports batch processing for multiple ensemble members simultaneously.
        """
        # Handle both single and batch inputs
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            dx = dx.unsqueeze(0) if dx.dim() == 2 else dx
            dy = dy.unsqueeze(0) if dy.dim() == 2 else dy
            single_input = True
        else:
            single_input = False
        
        batch_size, _, ny, nx = data.shape
        
        # Create coordinate grids
        y_coords = torch.arange(ny, device=self.device).view(1, -1, 1).expand(batch_size, ny, nx)
        x_coords = torch.arange(nx, device=self.device).view(1, 1, -1).expand(batch_size, ny, nx)
        
        # Calculate new coordinates
        new_y_coords = y_coords.float() + dy
        new_x_coords = x_coords.float() + dx
        
        # Normalize to [-1, 1] for grid_sample
        new_y_coords = 2.0 * new_y_coords / (ny - 1) - 1.0
        new_x_coords = 2.0 * new_x_coords / (nx - 1) - 1.0
        
        # Stack to create grid [B, H, W, 2]
        grid = torch.stack([new_x_coords, new_y_coords], dim=-1)
        
        # Apply displacement using grid_sample (bilinear interpolation)
        displaced_data = F.grid_sample(data, grid, mode='bilinear', 
                                      padding_mode='reflection', align_corners=True)
        
        # Handle land mask
        land_mask_expanded = land_mask.unsqueeze(0).unsqueeze(0).expand_as(data)
        invalid_mask = land_mask_expanded | torch.isnan(displaced_data)
        displaced_data = torch.where(invalid_mask, data, displaced_data)
        
        if single_input:
            displaced_data = displaced_data.squeeze(0).squeeze(0)
        
        return displaced_data
    
    def displace_for_channels_gpu_batch(self,
                                        dx_t0: torch.Tensor,
                                        dy_t0: torch.Tensor,
                                        dx_t1: torch.Tensor,
                                        dy_t1: torch.Tensor,
                                        batch_size: int = 1) -> xr.DataArray :
        """
        GPU-accelerated channel displacement with batch processing.
        Process multiple ensemble members simultaneously.
        """
        num_channels = self.dataset.isel(time=1).sizes['ch']
        
        # Expand displacement fields for batch processing
        if dx_t0.dim() == 2 :
            dx_t0 = dx_t0.unsqueeze(0).expand(batch_size, -1, -1)
            dy_t0 = dy_t0.unsqueeze(0).expand(batch_size, -1, -1)
            dx_t1 = dx_t1.unsqueeze(0).expand(batch_size, -1, -1)
            dy_t1 = dy_t1.unsqueeze(0).expand(batch_size, -1, -1)
        
        displaced_datarray_t0 = []
        displaced_datarray_t1 = []
        
        # Process channels in batches
        for ch in range(num_channels) :
            # Load data to GPU
            dataarray_2d_t0 = self.dataset["data"].isel(time=0, ch=ch)
            dataarray_2d_t1 = self.dataset["data"].isel(time=1, ch=ch)
            
            # Compute Dask arrays and move to GPU
            data_t0 = torch.from_numpy(dataarray_2d_t0.compute().values).float().to(self.device)
            data_t1 = torch.from_numpy(dataarray_2d_t1.compute().values).float().to(self.device)
            
            # Create land mask
            land_mask = self.create_land_mask_gpu(dataarray_2d_t0, device=self.device)
            
            # Batch displacement
            data_t0_batch = data_t0.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            data_t1_batch = data_t1.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            
            displaced_t0 = self.displace_array_gpu(data_t0_batch, dx_t0, dy_t0, land_mask)
            displaced_t1 = self.displace_array_gpu(data_t1_batch, dx_t1, dy_t1, land_mask)
            
            # Move back to CPU and convert to numpy
            displaced_datarray_t0.append(displaced_t0.squeeze(1).cpu().numpy())
            displaced_datarray_t1.append(displaced_t1.squeeze(1).cpu().numpy())
        
        # Stack and create DataArray
        displaced_t0_stacked = np.stack(displaced_datarray_t0, axis=1)  # [batch, ch, lat, lon]
        displaced_t1_stacked = np.stack(displaced_datarray_t1, axis=1)
        
        # Combine time dimension - stack along new time axis
        # Result shape: [batch, time=2, ch, lat, lon]
        displaced_combined = np.stack([displaced_t0_stacked, displaced_t1_stacked], axis=1)
        
        return displaced_combined
    
    def generate_batch_perturbations(self,
                                    perturbation_type: str,
                                    seeds: List[int],
                                    is_same_noise_in_time: bool = True,
                                    is_same_noise_for_component: bool = True,
                                    extraction_depth: int = 0) -> List[np.ndarray] :
        """
        Generate multiple perturbations in parallel on GPU.
        
        Args:
            perturbation_type: 'one_vector' or 'pointwise'
            seeds: List of seeds for ensemble members
            
        Returns:
            List of perturbed states (as numpy arrays for saving)
        """
        batch_size = min(len(seeds), self.batch_size)
        is_pointwise = perturbation_type != 'one_vector'
        
        results = []
        
        # Process in batches
        for batch_start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[batch_start:batch_start + batch_size]
            current_batch_size = len(batch_seeds)
            
            # Generate noise for batch
            dx_t0_batch = []
            dy_t0_batch = []
            dx_t1_batch = []
            dy_t1_batch = []
            
            for seed in batch_seeds :
                # Select amplitude based on perturbation type
                amplitude = self.onenoise_amplitude if perturbation_type == 'one_vector' else self.pointwisenoise_amplitude
                
                if is_same_noise_for_component :
                    if is_same_noise_in_time:
                        noise_gen = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, seed, self.device, amplitude)
                        noise_u_t0 = noise_gen.noise
                        noise_v_t0, noise_u_t1, noise_v_t1 = noise_u_t0, noise_u_t0, noise_u_t0
                    else :
                        noise_gen_t0 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                   seed, self.device, amplitude)
                        noise_gen_t1 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                   seed + 1, self.device, amplitude)
                        noise_u_t0 = noise_gen_t0.noise
                        noise_v_t0 = noise_u_t0
                        noise_u_t1 = noise_gen_t1.noise
                        noise_v_t1 = noise_u_t1
                else :
                    if is_same_noise_in_time :
                        noise_u_t0 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed, self.device, amplitude).noise
                        noise_v_t0 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed + 1, self.device, amplitude).noise
                        noise_u_t1, noise_v_t1 = noise_u_t0, noise_v_t0
                    else :
                        noise_u_t0 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed, self.device, amplitude).noise
                        noise_v_t0 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed + 1, self.device, amplitude).noise
                        noise_u_t1 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed + 2, self.device, amplitude).noise
                        noise_v_t1 = GPUStochasticNoiseGenerator(self.ssh_anomaly, is_pointwise, 
                                                                 seed + 3, self.device, amplitude).noise
                
                # Compute displacement fields
                if perturbation_type == 'one_vector' :
                    # Use min/max values
                    depth = extraction_depth
                    u_val_t0 = self.uv_depth_t0[f"max_d{depth}"].isel(ch=0).compute().values
                    v_val_t0 = self.uv_depth_t0[f"max_d{depth}"].isel(ch=1).compute().values
                    u_val_t1 = self.uv_depth_t1[f"max_d{depth}"].isel(ch=0).compute().values
                    v_val_t1 = self.uv_depth_t1[f"max_d{depth}"].isel(ch=1).compute().values
                    
                    grid_shape = (self.uv_depth_t1[f"d{depth}"].sizes['lat'], 
                                 self.uv_depth_t1[f"d{depth}"].sizes['lon'])
                    u_2d_t0 = torch.full(grid_shape, float(u_val_t0), device=self.device)
                    v_2d_t0 = torch.full(grid_shape, float(v_val_t0), device=self.device)
                    u_2d_t1 = torch.full(grid_shape, float(u_val_t1), device=self.device)
                    v_2d_t1 = torch.full(grid_shape, float(v_val_t1), device=self.device)
                    
                    dx_t0 = self.dt * (u_2d_t0 + noise_u_t0)
                    dy_t0 = -1 * self.dt * (v_2d_t0 + noise_v_t0)
                    dx_t1 = self.dt * (u_2d_t1 + noise_u_t1)
                    dy_t1 = -1 * self.dt * (v_2d_t1 + noise_v_t1)
                    
                else :  # pointwise
                    depth = 0
                    u_t0 = torch.from_numpy(self.uv_depth_t0[f"d{depth}"].isel(ch=0).compute().values).float().to(self.device)
                    v_t0 = torch.from_numpy(self.uv_depth_t0[f"d{depth}"].isel(ch=1).compute().values).float().to(self.device)
                    u_t1 = torch.from_numpy(self.uv_depth_t1[f"d{depth}"].isel(ch=0).compute().values).float().to(self.device)
                    v_t1 = torch.from_numpy(self.uv_depth_t1[f"d{depth}"].isel(ch=1).compute().values).float().to(self.device)
                    
                    dx_t0 = self.dt * u_t0 * noise_u_t0
                    dy_t0 = -1 * self.dt * v_t0 * noise_v_t0
                    dx_t1 = self.dt * u_t1 * noise_u_t1
                    dy_t1 = -1 * self.dt * v_t1 * noise_v_t1
                
                dx_t0_batch.append(dx_t0)
                dy_t0_batch.append(dy_t0)
                dx_t1_batch.append(dx_t1)
                dy_t1_batch.append(dy_t1)
            
            # Stack batch
            dx_t0_stacked = torch.stack(dx_t0_batch)
            dy_t0_stacked = torch.stack(dy_t0_batch)
            dx_t1_stacked = torch.stack(dx_t1_batch)
            dy_t1_stacked = torch.stack(dy_t1_batch)
            
            # Apply displacement to all channels in batch
            perturbed_batch = self.displace_for_channels_gpu_batch(
                dx_t0_stacked, dy_t0_stacked, dx_t1_stacked, dy_t1_stacked, 
                current_batch_size
            )
            
            # Split batch results
            for i in range(current_batch_size):
                results.append(perturbed_batch[i:i+1])
        
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Ensemble Generation for Ocean Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', 
                        type=str,
                        default='calibrate',
                        choices=['calibrate', 'initialize'],
                        help='Calibrate or initialize the ensemble generation process')
    
    parser.add_argument('--data-path', 
                        type=str,
                        default='/Odyssey/public/glonet/glorys12_1993-01-01_to_1993-06-30_init_states/combined_input.nc',
                        help='Path to input NetCDF data file or directory. In calibrate mode, reads all .nc files from directory')
    
    parser.add_argument('--num-perturbations',
                        type=int,
                        default=10,
                        help='Number of perturbations to generate per member (when using --input-dir)')
    
    parser.add_argument('--perturbation-type',
                        type=str,
                        default='pointwise',
                        choices=['one_vector', 'pointwise'],
                        help='Type of perturbation to generate')
    
    parser.add_argument('--target-idx', 
                        type=int, 
                        default=1, 
                        help='Target time index in the dataset')
    
    parser.add_argument('--climato-path', 
                        type=str, 
                        default='/Odyssey/public/glonet/1993-06-01_climatology/mean1.nc', 
                        help='Path to climatology data file')
    
    parser.add_argument('--rand-ds-path', 
                        type=str, 
                        default='/Odyssey/public/glonet/2022-06-01_init_states/input1.nc', 
                        help='Path to random dataset file')
    
    parser.add_argument('--output-dir', 
                        type=str, 
                        default=None,
                        help='Output directory for ensemble members')
    
    parser.add_argument('--dt', 
                        type=float, 
                        default=1.0, 
                        help='Scaling factor for displacement calculation (similiar to time step)')
    
    parser.add_argument('--onenoise-amplitude', 
                        type=float, 
                        default=0.1, 
                        help='Amplitude for one-vector noise generation')
    
    parser.add_argument('--pointwisenoise-amplitude', 
                        type=float, 
                        default=0.1, 
                        help='Amplitude for pointwise noise generation')
    
    parser.add_argument('--base-seed', 
                        type=int, 
                        default=42, 
                        help='Base random seed for reproducibility')
    
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=40,
                        help='Number of ensemble members to process in parallel on GPU')
    
    return parser.parse_args()


def main() :
    """Main execution function"""
    args = parse_args()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("GPU-ACCELERATED ENSEMBLE GENERATION")
    print("="*80)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Check mode and route accordingly
    if args.mode == 'calibrate':
        run_calibrate_mode(args, device)
    else:
        run_initialize_mode(args, device)


def run_calibrate_mode(args, device):
    """Run calibrate mode: read all NetCDF files from directory and generate perturbations"""
    data_path = Path(args.data_path)
    
    # Collect all NetCDF files from directory
    if data_path.is_dir():
        nc_files = sorted(list(data_path.glob("*.nc")))
        print(f"Found {len(nc_files)} NetCDF files in {data_path}")
    else:
        # Single file mode
        nc_files = [data_path]
        print(f"Processing single file: {data_path}")
    
    if not nc_files:
        print("Error: No NetCDF files found!")
        return
    
    # Create output directories
    if args.output_dir :
        perturbation_dir = Path(args.output_dir) 
    else :
        output_dir = Path(args.data_path)
        perturbation_dir = output_dir / args.perturbation_type

    
    perturbation_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nConfiguration:")
    print(f"  Mode: calibrate")
    print(f"  Data path: {args.data_path}")
    print(f"  Number of input files: {len(nc_files)}")
    print(f"  Perturbations per file: {args.num_perturbations}")
    print(f"  Perturbation type: {args.perturbation_type}")
    print(f"  Target index: {args.target_idx}")
    print(f"  Batch size: {args.batch_size} (parallel ensemble members)")
    print(f"  Output directory: {perturbation_dir}")
    print()
    
    start_time = time.time()
    total_members_generated = 0
    
    # Process each NetCDF file
    for file_idx, nc_file in enumerate(nc_files):
        print("="*80)
        print(f"Processing file {file_idx + 1}/{len(nc_files)}: {nc_file.name}")
        print("="*80)
        
        # Initialize generator for this file
        print("Initializing GPU generator...")
        file_start = time.time()
        generator = GPUCurrentBasedPerturbationGenerator(
            data_path=str(nc_file),
            target_idx=args.target_idx,
            climato_path=args.climato_path,
            rand_ds_path=args.rand_ds_path,
            dt=args.dt,
            device=device,
            batch_size=args.batch_size,
            onenoise_amplitude=args.onenoise_amplitude,
            pointwisenoise_amplitude=args.pointwisenoise_amplitude
        )
        print(f"✓ Initialization complete ({time.time() - file_start:.2f}s)\n")
        
        # Generate perturbations
        seeds = [args.base_seed + file_idx * 10000 + i for i in range(args.num_perturbations)]
        
        # Set perturbation parameters based on type
        if args.perturbation_type == 'one_vector':
            is_same_noise_in_time = True
            is_same_noise_for_component = True
        else:  # pointwise
            is_same_noise_in_time = False
            is_same_noise_for_component = False
        
        print(f"Generating {args.num_perturbations} perturbations using {args.perturbation_type}...") 
        perturbed_states = generator.generate_batch_perturbations(
            perturbation_type=args.perturbation_type,
            seeds=seeds,
            is_same_noise_in_time=is_same_noise_in_time,
            is_same_noise_for_component=is_same_noise_for_component,
            extraction_depth=0 if args.perturbation_type == 'one_vector' else None
        )
        
        # Save perturbations
        for idx, state in enumerate(perturbed_states):
            member_id = file_idx * args.num_perturbations + idx
            output_path = perturbation_dir / f"member_{member_id:03d}_initial_condition.nc"
            
            ds_out = xr.Dataset({
                "data": xr.DataArray(
                    state.squeeze(0),
                    dims=["time", "ch", "lat", "lon"],
                    coords={
                        "time": generator.dataset["data"].time,
                        "ch": generator.dataset["data"].ch,
                        "lat": generator.dataset["data"].lat,
                        "lon": generator.dataset["data"].lon
                    }
                )
            })
            ds_out.to_netcdf(output_path)
            total_members_generated += 1
        
        file_elapsed = time.time() - file_start
        print(f"✓ File {file_idx + 1} complete: {file_elapsed:.2f}s ({file_elapsed/args.num_perturbations:.2f}s per member)")
        print(f"  Saved {args.num_perturbations} members (total: {total_members_generated})\n")
    
    # Summary
    total_time = time.time() - start_time
    print("="*80)
    print("CALIBRATE MODE COMPLETE")
    print("="*80)
    print(f"Total files processed: {len(nc_files)}")
    print(f"Total members generated: {total_members_generated}")
    print(f"Total time: {total_time:.2f}s ({total_time/total_members_generated:.2f}s per member)")
    print(f"Output directory: {perturbation_dir}")
    print()


def run_initialize_mode(args, device):
    """Run initialize mode: generate ensemble with 40/80 members for two methods"""
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onevector_dir = output_dir / "one_vector"
    pointwise1_dir = output_dir / "pointwise"
    
    for dir_path in [onevector_dir, pointwise1_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Configuration:")
    print(f"  Mode: initialize")
    print(f"  Data path: {args.data_path}")
    print(f"  Target index: {args.target_idx}")
    print(f"  Batch size: {args.batch_size} (parallel ensemble members)")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Initialize GPU generator
    print("Initializing GPU data loader...")
    start_time = time.time()
    generator = GPUCurrentBasedPerturbationGenerator(
        data_path=args.data_path,
        target_idx=args.target_idx,
        climato_path=args.climato_path,
        rand_ds_path=args.rand_ds_path,
        dt=args.dt,
        device=device,
        batch_size=args.batch_size,
        onenoise_amplitude=args.onenoise_amplitude,
        pointwisenoise_amplitude=args.pointwisenoise_amplitude
    )
    print(f"✓ Initialization complete ({time.time() - start_time:.2f}s)")
    print()
    
    # Method 1: One Vector (50 members)
    print("="*80)
    print("METHOD 1: ONE VECTOR - 50 MEMBERS (GPU Parallel)")
    print("="*80)
    start_method1 = time.time()
    
    seeds_onevector = [args.base_seed + i + 10 for i in range(50)]
    perturbed_states = generator.generate_batch_perturbations(
        perturbation_type='one_vector',
        seeds=seeds_onevector,
        is_same_noise_in_time=True,
        is_same_noise_for_component=True,
        extraction_depth=0
    )
    
    # Save results
    for idx, state in enumerate(perturbed_states) :
        output_path = onevector_dir / f"member_{idx:03d}_initial_condition.nc"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ds_out = xr.Dataset({
            "data": xr.DataArray(
                state.squeeze(0),
                dims=["time", "ch", "lat", "lon"],
                coords={
                    "time": generator.dataset["data"].time,
                    "ch": generator.dataset["data"].ch,
                    "lat": generator.dataset["data"].lat,
                    "lon": generator.dataset["data"].lon
                }
            )
        })
        ds_out.to_netcdf(output_path)
        
        if (idx + 1) % 10 == 0 :
            print(f"  Saved {idx + 1}/50 members")
    
    elapsed_method1 = time.time() - start_method1
    print(f"✓ Method 1 complete: {elapsed_method1:.2f}s ({elapsed_method1/50:.2f}s per member)")
    print()
    
    # Method 2: Pointwise 1 (100 members)
    print("="*80)
    print("METHOD 2: POINTWISE 1 - 100 MEMBERS (GPU Parallel)")
    print("="*80)
    start_method2 = time.time()
    
    seeds_pointwise1 = [args.base_seed + 1000 + i for i in range(100)]
    perturbed_states = generator.generate_batch_perturbations(
        perturbation_type='pointwise',
        seeds=seeds_pointwise1,
        is_same_noise_in_time=False,
        is_same_noise_for_component=False
    )
    
    # Save results
    for idx, state in enumerate(perturbed_states):
        output_path = pointwise1_dir / f"member_{idx:03d}_initial_condition.nc"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ds_out = xr.Dataset({
            "data": xr.DataArray(
                state.squeeze(0),
                dims=["time", "ch", "lat", "lon"],
                coords={
                    "time": generator.dataset["data"].time,
                    "ch": generator.dataset["data"].ch,
                    "lat": generator.dataset["data"].lat,
                    "lon": generator.dataset["data"].lon
                }
            )
        })
        ds_out.to_netcdf(output_path)
        
        if (idx + 1) % 20 == 0:
            print(f"  Saved {idx + 1}/100 members")
    
    elapsed_method2 = time.time() - start_method2
    print(f"✓ Method 2 complete: {elapsed_method2:.2f}s ({elapsed_method2/100:.2f}s per member)")
    print()
    
    # Summary
    total_time = time.time() - start_time
    print("="*80)
    print("GPU-ACCELERATED ENSEMBLE GENERATION COMPLETE")
    print("="*80)
    print(f"Total time: {total_time:.2f}s ({total_time/150:.2f}s per member)")
    print(f"  - Method 1 (50 members): {elapsed_method1:.2f}s")
    print(f"  - Method 2 (100 members): {elapsed_method2:.2f}s")
    print()
    print(f"Speedup factor: Processing {args.batch_size} members in parallel")
    print()
    
    # Verify
    onevector_count = len(list(onevector_dir.glob("member_*_initial_condition.nc")))
    pointwise1_count = len(list(pointwise1_dir.glob("member_*_initial_condition.nc")))
    
    print("Verification:")
    print(f"  - One vector:  {onevector_count}/50")
    print(f"  - Pointwise 1: {pointwise1_count}/100")
    print(f"  - Total:       {onevector_count + pointwise1_count}/150")
    print()


if __name__ == '__main__':
    main()
