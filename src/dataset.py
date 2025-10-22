#!/usr/bin/env python3
"""
XrDataset class for GLONET using xarray for NetCDF data handling
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import xarray as xr
import dask.array as da
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import logging
log = logging.getLogger(__name__)


class XrDataset(Dataset):
    """
    Dataset class for handling NetCDF files using xarray for GLONET model.
    
    This dataset expects input data in the format [Time, Channel, Height, Width]
    and provides sequences for temporal forecasting.
    """
    
    # Class-level cache for shared data and preprocessing results
    _data_cache = {}
    _preprocessing_cache = {}
    
    def __init__(self, 
                 data_paths: Union[str, List[str], Dict[str, str]],
                 variables: Optional[List[str]] = None,
                 spatial_dims: Tuple[str, str] = ('lat', 'lon'),
                 time_dim: str = 'time',
                 time_minibatch_size: int = 15,
                 patch_size: Tuple[int, int] = (96, 96),
                 enable_time_minibatching: bool = True,  
                 enable_patching: bool = True, 
                 sequence_length: int = 2,
                 forecast_horizon: int = 7,
                 crop_zone: Optional[Tuple[int, int, int, int]] = None,
                 normalize: bool = False,
                 standardize: bool = True,
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 stat_path: str = "data/statistics.pth",
                 random_seed: int = 42,
                 shared_data: Optional[xr.Dataset] = None


    ):
        """
        Initialize XrDataset
        
        Args:
            data_paths: Path(s) to NetCDF file(s). Can be:
                - Single string path
                - List of paths 
                - Dict mapping variable names to paths
            variables: List of variable names to use. If None, uses all variables
            spatial_dims: Names of spatial dimensions (height, width)
            time_dim: Name of time dimension
            time_minibatch_size: Minibatch size for time dimension processing (matches time chunk size)
            patch_size: Spatial patch size (lat, lon) for spatial batching (matches spatial chunk size)
            enable_time_minibatching: Whether to enable time minibatching
            enable_patching: Whether to enable spatial patching
            sequence_length: Number of timesteps in input sequence
            forecast_horizon: Number of timesteps to forecast
            crop_zone: Spatial crop size (H1, W1, H2, W2). If None, uses full spatial extent
            normalize: Whether to normalize data to [0, 1]
            standardize: Whether to standardize data (mean=0, std=1)
            split: Dataset split ('train', 'val', 'test')
            split_ratios: Ratios for train/val/test split
            random_seed: Random seed for reproducible splits
            shared_data: Pre-loaded and preprocessed data to share across dataset instances

        """
        self.data_paths = data_paths
        self.variables = variables
        self.spatial_dims = spatial_dims
        self.time_dim = time_dim
        self.time_minibatch_size = time_minibatch_size
        self.patch_size = patch_size
        self.enable_time_minibatching = enable_time_minibatching
        self.enable_patching = enable_patching
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.crop_zone = crop_zone
        self.normalize = normalize
        self.standardize = standardize
        self.split = split
        self.split_ratios = split_ratios
        self.stat_path = stat_path
        self.random_seed = random_seed

        # Create a cache key for this configuration
        self.cache_key = self._create_cache_key()
        
        # Load and process data (using cache if available)
        if shared_data is not None:
            log.info("Using shared preprocessed data")
            log.info(f"    ====")
            self.data = shared_data
        else:
            self.data = self._get_or_load_data()
            self.data = self._get_or_preprocess_data()
        
        # Generate patch and minibatch indices
        self._generate_patch_indices()
        self._generate_time_minibatch_indices()
        
        # Create valid indices for forecasting
        self._make_valid_indices()
        self._split_indices()
        
        # Calculate statistics for normalization
        if self.normalize or self.standardize :
            self._calculate_statistics()
        
        log.info(f"Initialized XrDataset with {len(self)} samples for split '{self.split}'")
        log.info(f"Data shape: {dict(self.data.sizes) if hasattr(self.data, 'dims') else 'N/A'}")
        log.info(f"    ====    ====    ==== Dataset instance {self.split} created")
        
    def _create_cache_key(self) -> str:
        """Create a unique cache key based on data loading configuration"""
        import hashlib
        
        # Include all parameters that affect data loading and preprocessing
        key_data = {
            'data_paths': str(self.data_paths),
            'variables': str(self.variables),
            'crop_zone': str(self.crop_zone),
            'time_minibatch_size': self.time_minibatch_size,
            'patch_size': self.patch_size,
        }
        
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_or_load_data(self) -> xr.Dataset:
        """Get data from cache or load it if not cached"""
        if self.cache_key not in self._data_cache:
            log.info("Loading data (not in cache)")
            self._data_cache[self.cache_key] = self._load_data()
        else:
            log.info("Using cached data")
        return self._data_cache[self.cache_key]
    
    def _get_or_preprocess_data(self) -> xr.Dataset:
        """Get preprocessed data from cache or preprocess it if not cached"""
        preprocess_key = f"{self.cache_key}_preprocessed"
        if preprocess_key not in self._preprocessing_cache:
            log.info("Preprocessing data (not in cache)")
            self._preprocessing_cache[preprocess_key] = self._preprocess_data()
        else:
            log.info("Using cached preprocessed data")
        return self._preprocessing_cache[preprocess_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear the data cache to free memory"""
        cls._data_cache.clear()
        cls._preprocessing_cache.clear()
        log.info("Cleared XrDataset cache")
    
    def _load_data(self) -> xr.Dataset :
        """Load data from NetCDF files with dask chunking"""
        
        # Define chunk sizes
        chunks = {'time': self.time_minibatch_size, 'lat': self.patch_size[0], 'lon': self.patch_size[1]}
        
        if isinstance(self.data_paths, str) :
            # Single file - use open_dataset with chunks
            data = xr.open_dataset(self.data_paths, chunks=chunks)
            
        elif isinstance(self.data_paths, list) :
            # Multiple files - use open_mfdataset with dask chunking
            data = xr.open_mfdataset(self.data_paths, 
                                   combine='by_coords',
                                   concat_dim=self.time_dim,
                                   chunks=chunks,
                                   parallel=True)
            
        elif isinstance(self.data_paths, dict) :
            # Dictionary mapping variables to files
            data_arrays = {}
            for var_name, path in self.data_paths.items():
                if isinstance(path, list):
                    # Multiple files for this variable
                    ds = xr.open_mfdataset(path, 
                                         combine='by_coords',
                                         concat_dim=self.time_dim,
                                         chunks=chunks,
                                         parallel=True)
                else:
                    # Single file for this variable
                    ds = xr.open_dataset(path, chunks=chunks)
                    
                if self.variables is None or var_name in self.variables:
                    data_arrays[var_name] = ds[var_name] if var_name in ds else ds[list(ds.data_vars)[0]]
            data = xr.Dataset(data_arrays)
            
        else:
            raise ValueError("data_paths must be str, list, or dict")
        
        # Select variables if specified
        if self.variables is not None :
            available_vars = list(data.data_vars)
            
            # selected_vars = []
            # for var in self.variables :
            #     if var in available_vars :
            #         selected_vars.append(var)
            selected_vars = [var for var in self.variables if var in available_vars]
            
            if not selected_vars : # len(selected_vars) == 0 :
                raise ValueError(f"None of the specified variables {self.variables} found in data. Available: {available_vars}")
            data = data[selected_vars]
        
        # Sort by time
        data = data.sortby(self.time_dim)
        
        log.info(f"Loaded data with variables: {list(data.data_vars)}")
        log.info(f"Time range: {data[self.time_dim].min().values} to {data[self.time_dim].max().values}")
        log.info(f"Spatial dimensions: {data[self.spatial_dims[0]].size} x {data[self.spatial_dims[1]].size}")
        log.info(f"Data chunks: {data.chunks}")
        log.info(f"    ====")
        
        return data
    
    
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

        # Rechunk after cropping to ensure chunk alignment with patch size
        chunks = {'time': self.time_minibatch_size, 'lat': self.patch_size[0], 'lon': self.patch_size[1]}
        self.data = self.data.chunk(chunks)
        
        log.info(f"Preprocessed data shape: {dict(self.data.sizes)}")
        log.info(f"Data chunks after preprocessing: {self.data.chunks}")
        log.info(f"    ====")        

        return self.data
    
    def _generate_patch_indices(self) -> None:
        """Generate spatial patch indices based on patch_size that align with chunk size"""
        
        if not self.enable_patching:
            self.patch_indices = [(0, 0)]  # Single patch covering full area
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

    def _generate_time_minibatch_indices(self) -> None:
        """Generate time minibatch indices based on time_minibatch_size that align with chunk size"""
        
        if not self.enable_time_minibatching:
            self.time_minibatch_indices = [0]  # Single minibatch covering full time
            self.num_time_minibatches = 1
            log.info("Time minibatching disabled - using full time extent")
            return
            
        time_size = self.data.sizes[self.time_dim]
        
        # Calculate number of time minibatches
        self.num_time_minibatches = time_size // self.time_minibatch_size
        
        # Generate time minibatch start indices
        self.time_minibatch_indices = []
        for i in range(self.num_time_minibatches):
            time_start = i * self.time_minibatch_size
            self.time_minibatch_indices.append(time_start)
        
        log.info(f"Generated {self.num_time_minibatches} time minibatches of size {self.time_minibatch_size}")
        log.info(f"Total time size: {time_size}, Time minibatch size: {self.time_minibatch_size}")
        log.info(f"    ====")
    

    def _make_valid_indices(self) :
        """Create valid indices combining time, patches, and time minibatches for forecasting"""
        
        T = self.data.sizes[self.time_dim]

        # We need at least sequence_length + forecast_horizon timesteps
        min_length = self.sequence_length + self.forecast_horizon
        
        if T < min_length:
            raise ValueError(f"Dataset too small: {T} timesteps, need at least {min_length}")

        # Generate valid time indices within each time minibatch
        self.valid_indices = []
        
        for minibatch_idx, time_minibatch_start in enumerate(self.time_minibatch_indices):
            # Check if this time minibatch has enough timesteps
            time_minibatch_end = min(time_minibatch_start + self.time_minibatch_size, T)
            minibatch_length = time_minibatch_end - time_minibatch_start
            
            if minibatch_length >= min_length:
                # Valid start indices within this time minibatch
                for time_start in range(time_minibatch_start, time_minibatch_end - min_length + 1):
                    for patch_idx, (lat_start, lon_start) in enumerate(self.patch_indices):
                        # Create combined index: (minibatch_idx, time_start, patch_idx, lat_start, lon_start)
                        self.valid_indices.append((minibatch_idx, time_start, patch_idx, lat_start, lon_start))
        
        log.info(f"Generated {len(self.valid_indices)} valid samples:")
        log.info(f"Time minibatches: {self.num_time_minibatches}")
        log.info(f"Spatial patches: {self.num_patches}")
        log.info(f"Valid time steps per minibatch: varies")
        log.info(f"    ====")
    
        
    def _split_indices(self) -> None :
        """Shuffle and split indices for train/val/test sets instead of directly shuffling the dataset."""
        
        T = len(self.valid_indices)
        train_size = int(T * self.split_ratios[0])
        val_size = int(T * self.split_ratios[1])
        test_size = T - train_size - val_size

        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        indices = np.random.permutation(T)

        if self.split == 'train' :
            self.split_indices = indices[:train_size]
        elif self.split == 'val' :
            self.split_indices = indices[train_size:train_size + val_size]
        elif self.split == 'test' :
            self.split_indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"Unknown split mode: {self.split}")

        # Sort indices to maintain temporal order within split
        self.split_indices = np.sort(self.split_indices)

        log.info(f"Split indices for {self.split} set:")
        log.info(f"    {self.split_indices} -- {len(self.split_indices)} samples")

    def _calculate_statistics(self) -> None :
        """Calculate mean and std for normalization/standardization"""
        
        if self.split == 'train' :
            # Calculate statistics only on training data across spatial dimensions
            # Dataset has (time, ch, lat, lon) dimensions with one variable 'data'
            
            # Dimension to make statistics (across time and spatial dimensions, keeping channel dimension)
            dim_stat = [self.time_dim, self.spatial_dims[0], self.spatial_dims[1]]
            
            # Compute statistics for each channel
            means = self.data.mean(dim=dim_stat, skipna=True).compute()
            stds = self.data.std(dim=dim_stat, skipna=True).compute()
            mins = self.data.min(dim=dim_stat, skipna=True).compute()
            maxs = self.data.max(dim=dim_stat, skipna=True).compute()

            self.means = means
            self.stds = stds
            self.data_mins = mins
            self.data_maxs = maxs
            
            # Ensure no zero std values and handle NaN in std
            # Iterate over the Dataset variables (should be just 'data')
            for var_name in self.data.data_vars:
                std_vals = self.stds[var_name]
                std_vals = xr.where(std_vals < 1e-8, 1e-8, std_vals)
                self.stds[var_name] = std_vals
            
            log.info(f"Calculated statistics for channels:")
            for var_name in self.data.data_vars:
                log.info(f"{var_name} - Mean: {self.means[var_name].values}, Std: {self.stds[var_name].values}")
                log.info(f"{var_name} - Range: [{self.data_mins[var_name].values}, {self.data_maxs[var_name].values}]")

            # Save statistics
            statistics = {
                'means': {var_name: self.means[var_name].values for var_name in self.data.data_vars},
                'stds': {var_name: self.stds[var_name].values for var_name in self.data.data_vars},
                'mins': {var_name: self.data_mins[var_name].values for var_name in self.data.data_vars},
                'maxs': {var_name: self.data_maxs[var_name].values for var_name in self.data.data_vars}
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.stat_path), exist_ok=True)
            torch.save(statistics, self.stat_path)
            log.info(f"Train statistics data saved to {self.stat_path}")

        else :
            # Load saved statistics data from training data
            if not os.path.exists(self.stat_path):
                raise FileNotFoundError(f"Statistics file not found: {self.stat_path}. "
                                      "Training dataset must be created first to generate statistics.")
            
            statistics = torch.load(self.stat_path)
            
            self.means = {var_name: statistics['means'][var_name] for var_name in statistics['means']}
            self.stds = {var_name: statistics['stds'][var_name] for var_name in statistics['stds']}
            self.data_mins = {var_name: statistics['mins'][var_name] for var_name in statistics['mins']}
            self.data_maxs = {var_name: statistics['maxs'][var_name] for var_name in statistics['maxs']}

            log.info(f"Loaded statistics for variables: {list(statistics['means'].keys())}")


    def __len__(self) -> int :
        """Return number of available sequences"""
        
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] :
        """
        Get a set of input sequences and its corresponding target

        Returns:
            input_sequence: [T, C, H, W] - input sequence (patch)
            target: [C, H, W] - target for forecasting (patch)
        """

        # Map split index to actual valid index
        actual_idx = self.split_indices[idx]
        minibatch_idx, time_start, patch_idx, lat_start, lon_start = self.valid_indices[actual_idx]
        
        # Extract spatial patch coordinates
        lat_dim, lon_dim = self.spatial_dims
        patch_lat, patch_lon = self.patch_size
        
        lat_end = lat_start + patch_lat
        lon_end = lon_start + patch_lon
        
        # Extract input sequence using isel for efficient dask array slicing
        input_sequence = self.data.isel({
            self.time_dim: slice(time_start, time_start + self.sequence_length),
            lat_dim: slice(lat_start, lat_end),
            lon_dim: slice(lon_start, lon_end)
        })
        
        # Extract target (forecast_horizon timesteps ahead)
        target_time_idx = time_start + self.sequence_length + self.forecast_horizon - 1
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
            var_input = input_sequence[var].compute().values
            var_target = target[var].compute().values
            
            # Apply normalization/standardization per variable
            if self.normalize:
                var_input = (var_input - self.data_mins[var]) / (self.data_maxs[var] - self.data_mins[var])
                var_target = (var_target - self.data_mins[var]) / (self.data_maxs[var] - self.data_mins[var])
            
            if self.standardize:
                # Broadcasting the mean/std properly across time and spatial dimensions
                mean_vals = self.means[var].values
                std_vals = self.stds[var].values
                
                var_input = (var_input - mean_vals) / std_vals
                var_target = (var_target - mean_vals) / std_vals
            
            input_arrays.append(var_input)
            target_arrays.append(var_target)
        
        # Stack variables along channel dimension
        # input_sequence: [T, C, H, W]  
        input_sequence = np.stack(input_arrays, axis=1) 
        # target: [C, H, W]
        target = np.stack(target_arrays, axis=0)
        
        # Convert to torch tensors
        input_sequence = torch.from_numpy(input_sequence).float()
        target = torch.from_numpy(target).float()
        
        return input_sequence, target
    
    
    def get_data_info(self) -> Dict :
        """Get information about the dataset"""
        return {
            'num_samples': len(self),
            'time_minibatch_size': self.time_minibatch_size,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches if hasattr(self, 'num_patches') else 0,
            'num_time_minibatches': self.num_time_minibatches if hasattr(self, 'num_time_minibatches') else 0,
            'enable_patching': self.enable_patching,
            'enable_time_minibatching': self.enable_time_minibatching,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'data_shape': dict(self.data.sizes),
            'num_channels': len(self.data.data_vars),
            'variables': list(self.data.data_vars),
            'chunks': dict(self.data.chunks) if hasattr(self.data, 'chunks') else None,
            'normalized': self.normalize,
            'standardized': self.standardize,
            'split': self.split,
        }


class GlonetDataModule(pl.LightningDataModule):
    """DataModule for GLONET using XrDataset"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Extract configuration parameters
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model

        # Dataset parameters
        self.dataset_params = {
            'data_paths' : self.data_cfg.get('data_paths', 'data/input.nc'),
            'variables' : self.data_cfg.get('variables', None),
            'spatial_dims' : self.data_cfg.get('dimensions', {}).get('spatial', ['lat', 'lon']),
            'time_dim' : self.data_cfg.get('dimensions', {}).get('time', 'time'),
            'time_minibatch_size': self.data_cfg.get('computing', {}).get('time_minibatch_size', 15),
            'patch_size': tuple(self.data_cfg.get('computing', {}).get('patch_size', [96, 96])),
            'enable_time_minibatching': self.data_cfg.get('computing', {}).get('enable_time_minibatching', True),
            'enable_patching': self.data_cfg.get('computing', {}).get('enable_patching', True),
            'sequence_length' : self.data_cfg.get('sequence_length', self.model_cfg.dim[0] 
                                                  if hasattr(self.model_cfg, 'dim') else 2),
            'forecast_horizon' : self.data_cfg.get('forecast_horizon', 10),
            'crop_zone' : self.data_cfg.get('preprocessing', {}).get('crop_zone', None),
            'normalize' : self.data_cfg.get('preprocessing', {}).get('normalize', True),
            'standardize' : self.data_cfg.get('preprocessing', {}).get('standardize', True),
            'split_ratios' : (
                self.data_cfg.get('train_split', 0.8),  
                self.data_cfg.get('val_split', 0.1),  
                self.data_cfg.get('test_split', 0.1) 
            ), 
            'stat_path': self.data_cfg.get('statistics', {}).get('stat_path', 'data/statistics.pth'),
            'random_seed': self.cfg.get('seed', 42),
        } 
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage.
        1. Avoid val_dataloader issue with pytorch lightning sanity check,
            create train dataset first to load data, before trainer.fit algorithm.
        2. Load data once and share it across all dataset instances to reduce redundancy.
        """
        if stage == 'fit' or stage is None:
            # Create training dataset first to load and preprocess data
            self.train_dataset = XrDataset(split='train', **self.dataset_params)
            
            # Create validation dataset sharing the same preprocessed data
            self.val_dataset = XrDataset(split='val', shared_data=self.train_dataset.data, **self.dataset_params)
            
        if stage == 'test' or stage is None:
            # Ensure training dataset exists first for data loading
            if not hasattr(self, 'train_dataset'):
                self.train_dataset = XrDataset(split='train', **self.dataset_params)
            
            # Create test dataset sharing the same preprocessed data  
            self.test_dataset = XrDataset(split='test', shared_data=self.train_dataset.data, **self.dataset_params)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          shuffle=True, 
                          num_workers=self.data_cfg.get('dataloader', {}).get('num_workers', 4))
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          shuffle=False, 
                          num_workers=self.data_cfg.get('dataloader', {}).get('num_workers', 4))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          shuffle=False, 
                          num_workers=self.data_cfg.get('dataloader', {}).get('num_workers', 4))
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up resources after training/testing"""
        # Clear the cache to free memory
        XrDataset.clear_cache()
        log.info("DataModule teardown completed")


