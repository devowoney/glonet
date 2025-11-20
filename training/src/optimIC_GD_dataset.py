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
        
        # Get the specific sample
        self.input_sequence, self.target, self.selected_date, self.land_mask = self._get_sample()
        
        # Standardization
        self.mean, self.std = self._standardize()
        self.stat_path = None

        
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
        
        if not self.enable_patching:
            self.patch_indices  = [(0, 0)]  # Single patch covering full area
            self.num_patches = 1
            log.info("[!!]Spatial patching disabled - using full spatial extent")
            return
            
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
        
        mean = torch.nan_to_num(input_sequence_persisted.mean(dim=dim_stat, keepdim=True))
        std = torch.nan_to_num(input_sequence_persisted.std(dim=dim_stat, keepdim=True))
        min = torch.nan_to_num(input_sequence_persisted.min(dim=dim_stat, keepdim=True))
        max = torch.nan_to_num(input_sequence_persisted.max(dim=dim_stat, keepdim=True))
        
        return mean, std, min, max

    def __len__(self) -> int :
        return len(self.valid_indices)
    
    def __getitem__(self, 
                    idx : int) -> Tuple[torch.Tensor, torch.Tensor] :
        """
        Get a set of input sequences and its corresponding target

        Returns:
            input_sequence: [T, C, H, W] - input sequence (patch)
            target: [C, H, W] - target for forecasting (patch)
        """

        # Map split index to actual valid index
        actual_idx = self.valid_indices[idx]
        patch_idx, lat_start, lon_start = self.valid_indices[actual_idx]
        
        # Extract spatial patch coordinates
        lat_dim, lon_dim = self.spatial_dims
        patch_lat, patch_lon = self.patch_size
        
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
        
        # Handle NaN values that might arise from division
        input_sequence = np.nan_to_num(input_sequence, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        # Convert to torch tensors
        input_sequence = torch.from_numpy(input_sequence).float()
        target = torch.from_numpy(target).float()
        
        return input_sequence, target




class GlorysDataModule(pl.LightningDataModule) :
    def __init__(self, cfg) :
        super().__init__()
        
        # Extract configuration parameters
        self.data_cfg = cfg.data

        # Dataset parameters
        self.dataset_params = {
            'data_paths' : self.data_cfg.get('data_paths', 'data/input.nc'),
            'variables' : self.data_cfg.get('variables', None),
            'spatial_dims' : self.data_cfg.get('dimensions', {}).get('spatial', ['lat', 'lon']),
            'time_dim' : self.data_cfg.get('dimensions', {}).get('time', 'time'),
            'patch_size': tuple(self.data_cfg.get('computing', {}).get('patch_size', [96, 96])),
            'enable_patching': self.data_cfg.get('computing', {}).get('enable_patching', True),
            'sequence_length' : self.data_cfg.get('sequence_length', 2),
            'forecast_horizon' : self.data_cfg.get('forecast_horizon', 10),
            'crop_zone' : self.data_cfg.get('preprocessing', {}).get('crop_zone', None),
            'normalize' : self.data_cfg.get('preprocessing', {}).get('normalize', True),
            'standardize' : self.data_cfg.get('preprocessing', {}).get('standardize', True),
            'random_seed': self.cfg.get('seed', 42),
            'batch_size': self.data_cfg.get('dataloader', {}).get('batch_size', 8),  # For chunking alignment
        } 

    def setup(self) :
        """Setup"""
        
        # Create training dataset for optimizing initial condition
        ds_params = self.dataset_params.copy()
        self.train_dataset = OptimizeInitialConditionDataset(**ds_params)

    def train_dataloader(self) -> DataLoader :
        
        self.setup()
        
        return DataLoader(self.train_dataset, 
                          shuffle=True, 
                          batch_size=self.data_cfg.get('dataloader', {}).get('batch_size', 1),
                          num_workers=self.data_cfg.get('dataloader', {}).get('num_workers', 4),
                          pin_memory=self.data_cfg.get('dataloader', {}).get('pin_memory', True))

    def val_dataloader(self) -> DataLoader :
        return None  # No validation for single sample optimization 