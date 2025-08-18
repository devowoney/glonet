#!/usr/bin/env python3
"""
XrDataset class for GLONET using xarray for NetCDF data handling
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
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
    
    def __init__(self, 
                 data_paths: Union[str, List[str], Dict[str, str]],
                 variables: Optional[List[str]] = None,
                 spatial_dims: Tuple[str, str] = ('lat', 'lon'),
                 time_dim: str = 'time',
                 sequence_length: int = 2,
                 forecast_horizon: int = 7,
                 crop_zone: Optional[Tuple[int, int, int, int]] = None,
                 normalize: bool = True,
                 standardize: bool = False,
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 stat_path: str = "data/statistics.pth",
                 random_seed: int = 42
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
            sequence_length: Number of timesteps in input sequence
            forecast_horizon: Number of timesteps to forecast
            crop_zone: Spatial crop size (H1, W1, H2, W2). If None, uses full spatial extent
            normalize: Whether to normalize data to [0, 1]
            standardize: Whether to standardize data (mean=0, std=1)
            split: Dataset split ('train', 'val', 'test')
            split_ratios: Ratios for train/val/test split
            random_seed: Random seed for reproducible splits
        """
        self.data_paths = data_paths
        self.variables = variables
        self.spatial_dims = spatial_dims
        self.time_dim = time_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.crop_zone = crop_zone
        self.normalize = normalize
        self.standardize = standardize
        self.split = split
        self.split_ratios = split_ratios
        self.stat_path = stat_path
        self.random_seed = random_seed
        
        # Load and process data
        self.data = self._load_data()
        self.data = self._preprocess_data()
        self.data = self._split_data()
        
        # Calculate statistics for normalization
        if self.normalize or self.standardize :
            self._calculate_statistics()
        
        # Create valid indices for sequences
        self._create_sequence_indices()
        
        log.info(f"Initialized XrDataset with {len(self)} samples for split '{split}'")
        log.info(f"Data shape: {self.data.shape}")
    
    def _load_data(self) -> xr.Dataset :
        """Load data from NetCDF files"""
        
        if isinstance(self.data_paths, str) :
            # Single file
            data = xr.open_dataset(self.data_paths)
            
        elif isinstance(self.data_paths, list) :
            # Multiple files - concatenate along time dimension
            datasets = [xr.open_dataset(path) for path in self.data_paths]
            data = xr.concat(datasets, dim=self.time_dim)
            
        elif isinstance(self.data_paths, dict) :
            # Dictionary mapping variables to files
            data_arrays = {}
            for var_name, path in self.data_paths.items():
                ds = xr.open_dataset(path)
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
        
        return data
    
    
    def _preprocess_data(self) -> torch.Tensor :
        """Preprocess the loaded data"""
        
        # Convert to numpy array
        # Stack variables along channel dimension
        data_arrays = []
        for var in self.data.data_vars :
            arr = self.data[var].values
            # Ensure we have the right dimensions [T, C, H, W]
            if len(arr.shape) == 4 :
                data_arrays.append(arr)
            else :
                raise ValueError(f"Expected 4D array for variable {var}, got shape {arr.shape}")

        # Stack along channel dimension: [T, C, H, W]
        data = np.stack(data_arrays, axis=1)
        
        # Spatial cropping if specified
        if self.crop_zone is not None :
            start_h, start_w, end_h, end_w = self.crop_zone
            
            # Center crop
            data = data[:, :, start_h:end_h, start_w:end_w]

        # Convert to torch tensor
        data = torch.from_numpy(data).float() #.double()
        
        log.info(f"Preprocessed data shape: {data.shape}")
        return data
    

    def _make_valid_indices(self) :
        """Create valid indices in taking into account `sequence_length` and `forecast_horizon`."""
        
        T = self.data.shape[0]

        # We need at least sequence_length + forecast_horizon timesteps
        min_length = self.sequence_length + self.forecast_horizon
        
        if T < min_length:
            raise ValueError(f"Dataset too small: {T} timesteps, need at least {min_length}")

        # Valid start indices
        self.valid_indices = list(range(T - min_length + 1))
        log.info(f"Checked {len(self.valid_indices)} valid sequence starting indices")
    
        
    def _split_indices(self) -> None :
        """Shuffle and split indices for train/val/test sets instead of directly shuffling the dataset."""
        
        T = len(self.valid_indices)
        train_size = int(T * self.split_ratio[0])
        val_size = int(T * self.split_ratio[1])
        test_size = T - train_size - val_size

        # Set random seed for reproducible splits
        np.random.seed(self.ran_seed)
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
        

    def _calculate_statistics(self) -> None :
        """Calculate mean and std for normalization/standardization"""
        
        if self.split == 'train' :
            # Calculate statistics only on training data
            self.mean = self.data.mean(dim=(0, 2, 3), keepdim=True)
            self.std = self.data.std(dim=(0, 2, 3), keepdim=True)
            
            # Avoid division by zero
            self.std = torch.clamp(self.std, min=1e-8)
            
            # For min-max normalization
            self.data_min = self.data.min()
            self.data_max = self.data.max()
            
            
            log.info(f"Calculated statistics - Mean: {self.mean.mean():.4f}, Std: {self.std.mean():.4f}")
            log.info(f"Data range: [{self.data_min:.4f}, {self.data_max:.4f}]")

            # Save statistics in tensor with pytorch
            
            statistics = {
                'data_mean' : torch.tensor([self.mean]),
                'data_std' : torch.tensor([self.std]),
                'data_min' : torch.tensor(self.data_min.item()),
                'data_max' : torch.tensor(self.data_max.item())
            }

            torch.save(statistics, self.stat_path)
            log.info(f"Train statistics data is save in {self.stat_path}")

        else :
            # Load saved statistics data from training data
            statistics = torch.load(self.stat_path)

            self.mean = statistics.get('data_mean')
            self.std = statistics.get('data_std')
            self.std = torch.clamp(self.std, min=1e-8)
            self.data_min = statistics.get('data_min')
            self.data_max = statistics.get('data_max')


    def __len__(self) -> int :
        """Return number of available sequences"""
        
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] :
        """
        Get a set of input sequences and its corresponding target

        Returns:
            input_sequence: [T, C, H, W] - input sequence
            target: [C, H, W] - target for forecasting
        """

        start_idx = self.valid_indices[idx] - 1

        # Extract input sequence
        input_sequence = self.data[start_idx:start_idx + self.sequence_length]
        
        # Extract target (forecast_horizon timesteps ahead)
        target_idx = start_idx + self.sequence_length + self.forecast_horizon - 1
        target = self.data[target_idx]
        
        # Apply normalization/standardization
        if self.normalize:
            input_sequence = (input_sequence - self.data_min) / (self.data_max - self.data_min)
            target = (target - self.data_min) / (self.data_max - self.data_min)
        
        if self.standardize:
            input_sequence = (input_sequence - self.mean) / self.std
            target = (target - self.mean.squeeze(0)) / self.std.squeeze(0)
        
        return input_sequence, target
    
    
    def get_data_info(self) -> Dict :
        """Get information about the dataset"""
        return {
            'num_samples': len(self),
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'data_shape': tuple(self.data.shape),
            'num_channels': self.data.shape[1],
            'normalized': self.normalize,
            'standardized': self.standardize,
            'split': self.split
        }


def create_xr_datasets(cfg) -> Tuple[XrDataset, XrDataset, XrDataset]:
    """
    Create train, validation, and test datasets from configuration
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Extract configuration parameters
    data_cfg = cfg.data
    model_cfg = cfg.model

    # Dataset parameters
    dataset_params = {
        'data_paths' : data_cfg.get('input_paths', 'data/input.nc'),
        'variables' : data_cfg.get('variables', None),
        'spatial_dims' : data_cfg.get('dimensions', {}).get('spatial', ['lat', 'lon']),
        'time_dim' : data_cfg.get('dimensions', {}).get('time', 'time'),
        'sequence_length' : data_cfg.get('sequence_length', model_cfg.dim[0] if hasattr(model_cfg, 'dim') else 2),
        'forecast_horizon' : data_cfg.get('forecast_horizon', 10),
        'crop_zone' : data_cfg.get('preprocessing', {}).get('crop_zone', None),
        'normalize' : data_cfg.get('preprocessing', {}).get('normalize', True),
        'standardize' : data_cfg.get('preprocessing', {}).get('standardize', True),
        'split_ratios' : (
            data_cfg.get('train_split', 0.8),  
            data_cfg.get('val_split', 0.1),  
            data_cfg.get('test_split', 0.1) 
        ), 
        'stat_path': data_cfg.get('statistics', {}).get('save_path', 'data/statistics.pth'),
        'random_seed': cfg.get('seed', 42) 
    } 
     
    # Create datasets for each split 
    train_dataset = XrDataset(split='train', **dataset_params)
    val_dataset = XrDataset(split='val', **dataset_params)
    test_dataset = XrDataset(split='test', **dataset_params)
    
    log.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset
