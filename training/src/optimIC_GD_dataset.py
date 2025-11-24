import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple, Union, Any
import xarray as xr
import pytorch_lightning as pl
import numpy as np

import logging
log = logging.getLogger(__name__)

class OptimizeInitialConditionDataset(torch.utils.data.Dataset) :
    """
    Return a specific sample from the dataset for optimizing initial conditions.
    Copy from dataset.py 
    """
        
    def __init__(self, 
                data_paths: Union[str, List[str], Dict[str, str]],
                variables: Optional[List[str]] = None,
                spatial_dims: Tuple[str, str] = ('lat', 'lon'),
                time_dim: str = 'time',
                patch_size: Tuple[int, int] = (96, 96),
                enable_patching: bool = True,
                sample_idx : int = 0,
                sequence_length: int = 2,
                forecast_horizon: int = 7,
                crop_zone: Optional[Tuple[int, int, int, int]] = None,
                normalize: bool = False,
                standardize: bool = True,
                # split: str = 'train',
                # split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                # stat_path: str = "data/statistics.pth",
                random_seed: int = 42,
                # shared_data: Optional[xr.Dataset] = None,
                batch_size: int = 8) :  # Added for chunking alignment

        self.data_paths = data_paths
        self.variables = variables
        self.spatial_dims = spatial_dims
        self.time_dim = time_dim
        self.patch_size = patch_size
        self.enable_patching = enable_patching
        self.sample_idx = sample_idx
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.crop_zone = crop_zone
        self.normalize = normalize
        self.standardize = standardize
        # self.split = split
        # self.split_ratios = split_ratios
        # self.stat_path = stat_path
        self.random_seed = random_seed
        self.batch_size = batch_size  # Store for chunking

        # Creating dataset 
        log.info(f"Creating XrDataset instance")
        self.data = self._load_data()  # Keep original xarray dataset
        self.data = self._preprocess_data()  # Convert to tensor
        
        # Generate patch indices
        self._generate_patch_indices()
        
        # Create valid indices regarding to input and forecast horizon
        self._make_valid_indices()
        
        # Calculate statistics
        self.mean, self.std, self.min, self.max = self._calculate_statistics()
        
        # Create ocean masks for three different depth inputs (full spatial extent)
        self._create_full_ocean_masks()
        
    def _load_data(self) -> xr.Dataset :
        """Load netcdf file using dask array"""
        

        try :
            chunks = {'time': self.batch_size, 'lat': self.patch_size[0], 'lon': self.patch_size[1]}
            dataset = xr.open_dataset(self.data_paths, chunks=chunks)
            log.info(f"Loaded {self.data_paths} successfully")
        except FileNotFoundError as e:
            print(f"Error loading dataset {self.data_paths}: {e}")
            raise
        
        return dataset
    
    def _preprocess_data(self) -> xr.Dataset :
        """Preprocess the loaded data while maintaining dask chunking"""
        
        # Spatial cropping if specified - apply to xarray dataset to maintain chunking
        if self.crop_zone is not None :
            start_h, start_w, end_h, end_w = self.crop_zone
            
            # Get spatial dimension names
            lat_dim, lon_dim = self.spatial_dims
            
            # Apply spatial cropping using xarray indexing to maintain dask arrays
            self.data = self.data.isel({
                lat_dim: slice(start_h, end_h),
                lon_dim: slice(start_w, end_w)
            })
            
            log.info(f"-->>Applied spatial cropping: {lat_dim}[{start_h}:{end_h}], {lon_dim}[{start_w}:{end_w}]")

        # Rechunk for efficient I/O - use batch_size for time chunking
        # This aligns chunking with actual DataLoader batch size for better I/O performance
        chunks = {'time': self.batch_size, 'lat': self.patch_size[0], 'lon': self.patch_size[1]}
        self.data = self.data.chunk(chunks)
        
        log.info(f"Preprocessed data shape: {dict(self.data.sizes)}")
        log.info(f"Data chunks after preprocessing: {self.data.chunks}")
        log.info(f"    ====")        

        return self.data
    
    def _generate_patch_indices(self) -> None:
        """Generate spatial patch indices based on patch_size that align with chunk size"""
        
        if not self.enable_patching :
            self.patch_indices  = [(0, 0)]  # Single patch covering full area
            self.num_patches = 1
            log.info("[!!]Spatial patching disabled - using full spatial extent")
            return
        
        # !!! In Inition Condition Optimization by Gradient Descent, we use global patching approach !!!
        # !!! But letting following code for future reference !!!
        lat_dim, lon_dim = self.spatial_dims
        lat_size = self.data.sizes[lat_dim]
        lon_size = self.data.sizes[lon_dim]
        
        patch_lat, patch_lon = self.patch_size
        
        # Calculate number of patches in each dimension
        num_lat_patches = lat_size // patch_lat
        num_lon_patches = lon_size // patch_lon
        
        # Generate patch start indices
        self.patch_indices = []
        for i in range(num_lat_patches):
            for j in range(num_lon_patches):
                lat_start = i * patch_lat
                lon_start = j * patch_lon
                self.patch_indices.append((lat_start, lon_start))
        
        self.num_patches = len(self.patch_indices)
        
        log.info(f"Generated {self.num_patches} spatial patches of size {self.patch_size}")
        log.info(f"Spatial coverage: {num_lat_patches}x{num_lon_patches} patches")
        log.info(f"Total spatial size: {lat_size}x{lon_size}, Patch size: {patch_lat}x{patch_lon}")
        log.info(f"    ====")        
        
    def _make_valid_indices(self) :
        """Create valid indices using global approach"""
        
        T = self.data.sizes[self.time_dim]

        # We need at least sequence_length + forecast_horizon timesteps
        min_length = self.sequence_length + self.forecast_horizon
        
        if T < min_length:
            raise ValueError(f"Dataset too small: {T} timesteps, need at least {min_length}")

        # Global approach: use all possible time positions
        self.valid_indices = []
        
        # for time_start in range(T - min_length + 1): # In optimization IC, we only need one sample
        for patch_idx, (lat_start, lon_start) in enumerate(self.patch_indices):
            # self.valid_indices.append((time_start, patch_idx, lat_start, lon_start))
            self.valid_indices.append((patch_idx, lat_start, lon_start))
            
        log.info(f"Generated {len(self.valid_indices)} valid samples (global indexing):")
        log.info(f"Spatial patches: {self.num_patches} for one time sample")
        log.info(f"    ====")

    def _calculate_statistics(self) -> None :
        
        dim_stat = [self.spatial_dims[0], self.spatial_dims[1]]
        input_sequence_persisted = self.data.isel(time=slice(self.sample_idx, self.sample_idx + self.sequence_length)).persist()
        
        self.mean = input_sequence_persisted.mean(dim=dim_stat, skipna=True)
        self.std = input_sequence_persisted.std(dim=dim_stat, skipna=True)
        self.min = input_sequence_persisted.min(dim=dim_stat, skipna=True)
        self.max = input_sequence_persisted.max(dim=dim_stat, skipna=True)
        
        # Ensure no zero std values and handle NaN in std
        # Iterate over the Dataset variables (should be just 'data')
        for var_name in self.data.data_vars:
            std_vals = self.std[var_name]
            std_vals = xr.where(std_vals < 1e-8, 1e-8, std_vals)
            self.std[var_name] = std_vals
                    
        return self.mean, self.std, self.min, self.max
    
    def _create_full_ocean_masks(self) -> None:
        """Create full spatial extent ocean masks for three different depth inputs based on NaN values.
        
        Creates three masks for the full spatial extent:
        - input1 (surface): channels 0-4
        - input2 (shallow): channels 5-44 (10 depth levels each of thetao, uo, vo)
        - input3 (deep): channels 45-84 (10 depth levels each of thetao, uo, vo)
        
        Ocean mask = 1 - land_mask, where land_mask is 1 for NaN locations.
        Stored as numpy arrays [C, H, W] for efficient patching in __getitem__.
        """
        log.info("Creating full ocean masks for three depth inputs...")
        
        # Get a sample time slice to extract spatial NaN pattern
        sample_data = self.data.isel({self.time_dim: self.sample_idx})
        
        # Extract data and convert to numpy
        data_values = sample_data['data'].persist().values  # Shape: [C, H, W]
        
        # Create land masks (1 where NaN, 0 where valid) for each channel
        land_mask = np.isnan(data_values).astype(np.float32)  # [C, H, W]
        
        # Split into three depth ranges and create ocean masks
        # input1: surface (channels 0-4)
        self.full_ocean_mask_1 = 1.0 - land_mask[0:5, :, :]  # [5, H, W]
        
        # input2: shallow (channels 5-44)
        self.full_ocean_mask_2 = 1.0 - land_mask[5:45, :, :]  # [40, H, W]
        
        # input3: deep (channels 45-84)
        self.full_ocean_mask_3 = 1.0 - land_mask[45:85, :, :]  # [40, H, W]
        
        log.info(f"Full ocean mask 1 (surface, ch 0-4) shape: {self.full_ocean_mask_1.shape}")
        log.info(f"Full ocean mask 2 (shallow, ch 5-44) shape: {self.full_ocean_mask_2.shape}")
        log.info(f"Full ocean mask 3 (deep, ch 45-84) shape: {self.full_ocean_mask_3.shape}")
        log.info(f"    ====")
    
    def _get_patch_ocean_masks(self, lat_start: int, lat_end: int, lon_start: int, lon_end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ocean masks for a specific spatial patch.
        
        Args:
            lat_start, lat_end, lon_start, lon_end: Patch coordinates
            
        Returns:
            Tuple of three ocean masks as torch tensors [C, H, W]
            These will auto-broadcast to [T, C, H, W] when multiplied with input sequences.
        """
        # Extract patches from full masks
        ocean_mask_1 = self.full_ocean_mask_1[:, lat_start:lat_end, lon_start:lon_end]
        ocean_mask_2 = self.full_ocean_mask_2[:, lat_start:lat_end, lon_start:lon_end]
        ocean_mask_3 = self.full_ocean_mask_3[:, lat_start:lat_end, lon_start:lon_end]
        
        # Convert to torch tensors
        ocean_mask_1 = torch.from_numpy(ocean_mask_1.copy()).float()
        ocean_mask_2 = torch.from_numpy(ocean_mask_2.copy()).float()
        ocean_mask_3 = torch.from_numpy(ocean_mask_3.copy()).float()
        
        return ocean_mask_1, ocean_mask_2, ocean_mask_3
    
    
    def __len__(self) -> int :
        return len(self.valid_indices)
    
    def __getitem__(self, 
                    idx : int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                        torch.Tensor, torch.Tensor, torch.Tensor] :
        """
        Get a set of input sequences and its corresponding target

        Returns:
            input_sequence: [T, C, H, W] - input sequence (patch)
            target: [C, H, W] - target for forecasting (patch)
        """

        # Map split index to actual valid index
        patch_idx, lat_start, lon_start = self.valid_indices[idx]
        
        # Extract spatial patch coordinates
        lat_dim, lon_dim = self.spatial_dims
        
        if self.enable_patching:
            patch_lat, patch_lon = self.patch_size
        else:
            patch_lat = self.data.sizes[lat_dim]
            patch_lon = self.data.sizes[lon_dim]
        
        lat_end = lat_start + patch_lat
        lon_end = lon_start + patch_lon
        
        # Extract input sequence using isel for efficient dask array slicing
        input_sequence = self.data.isel({
            self.time_dim: slice(self.sample_idx, self.sample_idx + self.sequence_length),
            lat_dim: slice(lat_start, lat_end),
            lon_dim: slice(lon_start, lon_end)
        })
        
        # Extract target (forecast_horizon timesteps ahead)
        target_time_idx = self.sample_idx + self.sequence_length + self.forecast_horizon - 1
        target = self.data.isel({
            self.time_dim: target_time_idx,
            lat_dim: slice(lat_start, lat_end),
            lon_dim: slice(lon_start, lon_end)
        })
        
        # Store coordinates for later use (before converting to numpy)
        self.current_coords = {
            'time': input_sequence[self.time_dim].values,
            'lat': input_sequence[lat_dim].values,
            'lon': input_sequence[lon_dim].values
        }
        
        # Convert to torch tensors and stack variables along channel dimension
        # Input sequence: [T, C, H, W]
        input_arrays = []
        target_arrays = []
        
        for var in self.data.data_vars:
            # Get input sequence for this variable and compute to load into memory
            var_input = input_sequence[var].persist().values
            var_target = target[var].persist().values
            
            # Apply normalization/standardization per variable
            if self.normalize and hasattr(self, 'data_mins') and hasattr(self, 'data_maxs'):
                # Handle both xarray and numpy array cases for loaded statistics
                data_min = self.min[var].values if hasattr(self.min[var], 'values') else self.min[var]
                data_max = self.max[var].values if hasattr(self.max[var], 'values') else self.max[var]
                
                # Reshape for proper broadcasting
                data_min_input = data_min.reshape(1, -1, 1, 1)   # (1, C, 1, 1) for broadcasting with (T, C, H, W)
                data_max_input = data_max.reshape(1, -1, 1, 1)
                data_min_target = data_min.reshape(-1, 1, 1)     # (C, 1, 1) for broadcasting with (C, H, W)
                data_max_target = data_max.reshape(-1, 1, 1)
                
                var_input = (var_input - data_min_input) / (data_max_input - data_min_input)
                var_target = (var_target - data_min_target) / (data_max_target - data_min_target)
            
            if self.standardize and hasattr(self, 'means') and hasattr(self, 'stds'):
                # Handle both xarray and numpy array cases for loaded statistics
                mean_vals = self.mean[var].values if hasattr(self.mean[var], 'values') else self.mean[var]
                std_vals = self.std[var].values if hasattr(self.std[var], 'values') else self.std[var]
                
                # Reshape for proper broadcasting: 
                # mean_vals/std_vals shape: (C,) -> (1, C, 1, 1) for input, (C, 1, 1) for target
                mean_vals_input = mean_vals.reshape(1, -1, 1, 1)  # (1, C, 1, 1) for broadcasting with (T, C, H, W)
                std_vals_input = std_vals.reshape(1, -1, 1, 1)
                mean_vals_target = mean_vals.reshape(-1, 1, 1)    # (C, 1, 1) for broadcasting with (C, H, W)
                std_vals_target = std_vals.reshape(-1, 1, 1)
                
                var_input = (var_input - mean_vals_input) / std_vals_input
                var_target = (var_target - mean_vals_target) / std_vals_target
            
            input_arrays.append(var_input)
            target_arrays.append(var_target)
        
        # Since data already has channel dimension [T, C, H, W], we concatenate along channel axis
        # input_sequence: [T, C, H, W]  
        input_sequence = np.concatenate(input_arrays, axis=1) if len(input_arrays) > 1 else input_arrays[0]
        # target: [C, H, W]
        target = np.concatenate(target_arrays, axis=0) if len(target_arrays) > 1 else target_arrays[0]
        
        # Make 3 divided arrays for 3 pretrained models 
        input_sequence_1 = input_sequence[:, :5, :, :]
        input_sequence_2 = input_sequence[:, 5:45, :, :]
        input_sequence_3 = input_sequence[:, 45:85, :, :]
        target_1 = target[:5, :, :]
        target_2 = target[5:45, :, :]
        target_3 = target[45:85, :, :]
        
        # Convert to torch tensors
        input_sequence_1 = torch.from_numpy(input_sequence_1).float()
        input_sequence_2 = torch.from_numpy(input_sequence_2).float()
        input_sequence_3 = torch.from_numpy(input_sequence_3).float()
        target_1 = torch.from_numpy(target_1).float()
        target_2 = torch.from_numpy(target_2).float()
        target_3 = torch.from_numpy(target_3).float()
        
        # Handle NaN values that might arise from division
        input_sequence_1 = torch.nan_to_num(input_sequence_1, nan=0.0)
        input_sequence_2 = torch.nan_to_num(input_sequence_2, nan=0.0)
        input_sequence_3 = torch.nan_to_num(input_sequence_3, nan=0.0)
        target_1 = torch.nan_to_num(target_1, nan=0.0)
        target_2 = torch.nan_to_num(target_2, nan=0.0)
        target_3 = torch.nan_to_num(target_3, nan=0.0)
        
        # Get ocean masks for this patch
        # Shape: [C, H, W] - will auto-broadcast to [T, C, H, W] when needed
        self.ocean_mask_1, self.ocean_mask_2, self.ocean_mask_3 = self._get_patch_ocean_masks(lat_start, 
                                                                                               lat_end, 
                                                                                               lon_start, 
                                                                                               lon_end)

        return input_sequence_1, input_sequence_2, input_sequence_3, target_1, target_2, target_3


class GlorysDataModule(pl.LightningDataModule) :
    def __init__(self, cfg) :
        super().__init__()
        
        # Extract configuration parameters
        self.cfg = cfg
        self.data_cfg = cfg.data

        # Dataset parameters
        self.dataset_params = {
            'data_paths' : self.data_cfg.get('data_paths', 'data/input.nc'),
            'variables' : self.data_cfg.get('variables', None),
            'spatial_dims' : self.data_cfg.get('dimensions', {}).get('spatial', ['lat', 'lon']),
            'time_dim' : self.data_cfg.get('dimensions', {}).get('time', 'time'),
            'patch_size': tuple(self.data_cfg.get('computing', {}).get('patch_size', [96, 96])),
            'enable_patching': self.data_cfg.get('computing', {}).get('enable_patching', False),
            'sample_idx' : self.data_cfg.get('sample_idx', 0),
            'sequence_length' : self.data_cfg.get('sequence_length', 2),
            'forecast_horizon' : self.data_cfg.get('forecast_horizon', 10),
            'crop_zone' : self.data_cfg.get('preprocessing', {}).get('crop_zone', None),
            'normalize' : self.data_cfg.get('preprocessing', {}).get('normalize', True),
            'standardize' : self.data_cfg.get('preprocessing', {}).get('standardize', True),
            'random_seed': self.cfg.get('seed', 42),
            'batch_size': self.data_cfg.get('dataloader', {}).get('batch_size', 8),  # For chunking alignment
        } 

    def setup(self, stage: Optional[str] = None) :
        """Setup - only creates train dataset for optimization"""
        
        # Create training dataset for optimizing initial condition
        ds_params = self.dataset_params.copy()
        self.train_dataset = OptimizeInitialConditionDataset(**ds_params)

    def train_dataloader(self) -> DataLoader :
        # For single-sample optimization, disable multiprocessing and shuffling
        # to avoid hanging issues
        num_samples = len(self.train_dataset)
        use_multiprocessing = num_samples > 1
        
        return DataLoader(self.train_dataset, 
                          shuffle=use_multiprocessing,  # No shuffle for single sample
                          batch_size=self.data_cfg.get('dataloader', {}).get('batch_size', 1),
                          num_workers=self.data_cfg.get('dataloader', {}).get('num_workers', 4) if use_multiprocessing else 0,
                          pin_memory=self.data_cfg.get('dataloader', {}).get('pin_memory', True))

    def val_dataloader(self) -> DataLoader :
        return None  # No validation for single sample optimization 