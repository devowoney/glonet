#!/usr/bin/env python3
"""
Example script demonstrating XrDataset usage
"""

import torch
import numpy as np
import xarray as xr
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from dataset import XrDataset

def create_sample_netcdf_data():
    """Create sample NetCDF data for testing"""
    # Create dummy ocean data
    np.random.seed(42)
    
    # Dimensions
    time_steps = 30
    lat_size = 64
    lon_size = 64
    
    # Create coordinates
    times = np.arange('2023-01-01', '2023-01-31', dtype='datetime64[D]')
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)
    
    # Create sample variables
    data_vars = {}
    
    # Sea surface temperature (thetao)
    thetao = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, time_steps))[:, None, None] + \
             np.random.randn(time_steps, lat_size, lon_size) * 0.5
    data_vars['thetao'] = (['time', 'lat', 'lon'], thetao)
    
    # Salinity (so)
    so = 35 + np.random.randn(time_steps, lat_size, lon_size) * 0.1
    data_vars['so'] = (['time', 'lat', 'lon'], so)
    
    # Eastward velocity (uo)
    uo = np.random.randn(time_steps, lat_size, lon_size) * 0.1
    data_vars['uo'] = (['time', 'lat', 'lon'], uo)
    
    # Northward velocity (vo)
    vo = np.random.randn(time_steps, lat_size, lon_size) * 0.1
    data_vars['vo'] = (['time', 'lat', 'lon'], vo)
    
    # Sea surface height (zos)
    zos = np.random.randn(time_steps, lat_size, lon_size) * 0.05
    data_vars['zos'] = (['time', 'lat', 'lon'], zos)
    
    # Create dataset
    coords = {'time': times, 'lat': lats, 'lon': lons}
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add attributes
    ds['thetao'].attrs = {'units': 'degrees_C', 'long_name': 'Sea Surface Temperature'}
    ds['so'].attrs = {'units': 'psu', 'long_name': 'Salinity'}
    ds['uo'].attrs = {'units': 'm/s', 'long_name': 'Eastward Velocity'}
    ds['vo'].attrs = {'units': 'm/s', 'long_name': 'Northward Velocity'}
    ds['zos'].attrs = {'units': 'm', 'long_name': 'Sea Surface Height'}
    
    return ds

def test_xr_dataset():
    """Test the XrDataset class"""
    # Create sample data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create and save sample data
    sample_file = data_dir / "sample_ocean.nc"
    if not sample_file.exists():
        print("Creating sample NetCDF data...")
        ds = create_sample_netcdf_data()
        ds.to_netcdf(sample_file)
        print(f"Sample data saved to {sample_file}")
    
    # Test XrDataset
    print("\nTesting XrDataset...")
    
    # Create dataset with specific parameters
    dataset = XrDataset(
        data_paths=str(sample_file),
        sequence_length=7,
        forecast_horizon=1,
        variables=['thetao', 'so', 'uo', 'vo', 'zos'],
        crop_size=(32, 32),
        normalize=True,
        standardize=False,
        split='train',
        split_ratios=(0.8, 0.1, 0.1)
    )
    
    print(f"Dataset info: {dataset.get_data_info()}")
    print(f"Number of samples: {len(dataset)}")
    
    # Get a sample
    input_seq, target = dataset[0]
    print(f"Input sequence shape: {input_seq.shape}")  # Should be [T, C, H, W]
    print(f"Target shape: {target.shape}")             # Should be [C, H, W]
    print(f"Input data range: [{input_seq.min():.3f}, {input_seq.max():.3f}]")
    print(f"Target data range: [{target.min():.3f}, {target.max():.3f}]")
    
    # Test with DataLoader
    from torch.utils.data import DataLoader
    
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"\nDataLoader test with {len(data_loader)} batches:")
    for i, (batch_input, batch_target) in enumerate(data_loader):
        print(f"Batch {i}: Input {batch_input.shape}, Target {batch_target.shape}")
        if i >= 2:  # Show first 3 batches
            break
    
    print("XrDataset test completed successfully!")

if __name__ == "__main__":
    test_xr_dataset()
