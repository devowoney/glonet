# XrDataset Integration for GLONET

## Overview
This document describes the XrDataset class created for GLONET and its integration into the training pipeline.

## What was implemented

### 1. XrDataset Class (`src/dataset.py`)
A comprehensive dataset class for handling NetCDF files using xarray, specifically designed for GLONET's temporal forecasting needs.

**Key Features:**
- **Multi-format support**: Single files, multiple files, or variable-to-file mappings
- **Flexible variable selection**: Choose specific variables from NetCDF files
- **Temporal sequences**: Creates input sequences of configurable length for forecasting
- **Data preprocessing**: Built-in normalization, standardization, and spatial cropping
- **Train/Val/Test splitting**: Automatic data splitting with configurable ratios
- **Memory efficient**: Uses xarray for efficient NetCDF handling

**Usage Examples:**
```python
# Single file with all variables
dataset = XrDataset(
    data_paths="data/ocean.nc",
    sequence_length=7,
    forecast_horizon=1,
    variables=['temperature', 'salinity'],
    normalize=True
)

# Multiple variables from different files
dataset = XrDataset(
    data_paths={
        'thetao': 'data/temperature.nc',
        'so': 'data/salinity.nc',
        'uo': 'data/u_velocity.nc',
        'vo': 'data/v_velocity.nc'
    },
    sequence_length=7,
    forecast_horizon=1
)
```

### 2. Updated Training Script (`train.py`)
Enhanced the training script to use XrDataset.

**Key Changes:**
- Added `create_data_loaders()` function that uses XrDataset
- Integrated with Hydra configuration system
- Proper device handling (CPU/CUDA) with automatic detection

### 3. Updated GLONET Model (`src/glonet.py`)
Modified the model to accept device parameter instead of hardcoding CUDA.

**Changes:**
- Added `device` parameter to constructor
- Removed hardcoded `.to('cuda:0')` calls
- Dynamic device assignment in forward pass

### 4. Configuration Updates
Updated configuration files to support the new dataset:

**Data Configuration (`config/data/default.yaml`):**
```yaml
# Input data paths - flexible format
input_paths: "data/sample_ocean.nc"

# Variables to use (null = all variables)
variables: null

# Dimension names
dimensions:
  time: "time"
  spatial: ["lat", "lon"]

# Preprocessing options
preprocessing:
  normalize: true
  standardize: true
  crop_size: [128, 128]

# Forecast parameters
sequence_length: 2      # Input sequence length
forecast_horizon: 7    # How many timesteps ahead to predict
```

## Data Format Requirements

The XrDataset expects NetCDF files with the following structure:
- **Time dimension**: Named 'time' (configurable)
- **Spatial dimensions**: Named 'lat' and 'lon' (configurable)
- **Variables**: Any number of data variables (e.g., 'thetao', 'so', 'uo', 'vo', 'zos')
- **Channel**: Copernicus Marine offers oceans depth data. Be aware that the input Dataset has flattened dimension with variables, called channel.

Example data structure (Copernicus Marine):
```
<xarray.Dataset>
Dimensions:  (time: 30, depth 10, lat: 64, lon: 64)
Coordinates:
  * time     (time) datetime64[ns] 2023-01-01 ... 2023-01-30
  * depth    (depth) float64 0.494 50.20 ... 785.40
  * lat      (lat) float64 -90.0 -87.1 ... 87.1 90.0
  * lon      (lon) float64 -180.0 -174.4 ... 174.4 180.0
Data variables:
    thetao   (time, lat, lon) float64 ...
    so       (time, lat, lon) float64 ...
    uo       (time, lat, lon) float64 ...
    vo       (time, lat, lon) float64 ...
```

Example data structure (GLONET input):
```
<xarray.Dataset>
Dimensions:  (time: 30, channel 40, lat: 64, lon: 64)
Coordinates:
  * time     (time) datetime64[ns] 2023-01-01 ... 2023-01-30
  * channel  (channel) int 1 2 ... 40
  * lat      (lat) float64 -90.0 -87.1 ... 87.1 90.0
  * lon      (lon) float64 -180.0 -174.4 ... 174.4 180.0
Data variables:
    thetao   (time, lat, lon) float64 ...
    so       (time, lat, lon) float64 ...
    uo       (time, lat, lon) float64 ...
    vo       (time, lat, lon) float64 ...

```

## Running the Training

### With Real Data
```bash
python train.py data.input_paths="path/to/your/data.nc"
```


### Configuration Override Examples
```bash
# Use specific data file
python train.py data.input_paths="data/my_ocean_data.nc"

# Select specific variables
python train.py data.variables="['thetao','so']"

# Change sequence length
python train.py data.forecast_cycle=14

# CPU training
python train.py device=cpu

# Quick test
python train.py training.epochs=1 training.batch_size=2
```

## Configuration Options

### Data Configuration
- `input_paths`: Path(s) to NetCDF files
- `variables`: List of variables to use (null for all)
-  `sequence_length`: Input sequence
-  `forecast_horizon`: Forecast time
- `preprocessing.crop_size`: Spatial cropping
- `preprocessing.normalize`: Min-max normalization
- `preprocessing.standardize`: Z-score standardization
- `train_split/val_split/test_split`: Data splitting ratios

### Model Configuration
- `dim`: [T, C, H, W] - model dimensions
- `device`: Device to use ('cpu', 'cuda', 'auto')

## Error Handling and Fallbacks

The system includes robust error handling:
1. **Missing data files**: Falls back to dummy data with warning
2. **Invalid NetCDF format**: Detailed error messages
3. **CUDA unavailable**: Automatic fallback to CPU
4. **Insufficient data**: Clear error messages about minimum requirements

## Next Steps

1. **Flexibility of CUDA**: Parallel computing
2. **Lightning**: Pytorch lightning integration

The system is now ready for production use with real oceanographic data while maintaining compatibility with the existing GLONET architecture.
