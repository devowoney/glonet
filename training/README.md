# GLONET Training Pipeline - Complete Guide

A comprehensive guide for training the GLONET ocean forecasting model from scratch.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Prepare Training Dataset](#prepare-training-dataset)
- [Environment Setup](#environment-setup)
- [Running Training](#running-training)
- [Training Monitor](#training-monitor)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

</br></br>



## Prerequisites

### 1. Hardware Requirements

**GPU Requirements:**
- **Minimum**: 60GB GPU memory
- **Recommended**: NVIDIA A100 (80GB) or equivalent
- CUDA-capable GPU (CUDA 11.0 or higher)

**System Requirements:**
- Linux operating system (tested on Ubuntu 20.04+)
- 64GB+ system RAM
- 500GB+ available disk space for data and checkpoints

### 2. Dataset Requirements

#### Dataset Format

GLONET accepts only NetCDF files with the following structure:

*Copernicus Marine Format (Raw Data)*
```
<xarray.Dataset>
Dimensions:  (time: 30, depth: 10, lat: 64, lon: 64)
Coordinates:
  * time     (time) datetime64[ns] 2023-01-01 ... 2023-01-30
  * depth    (depth) float64 0.494 50.20 ... 785.40
  * lat      (lat) float64 -90.0 -87.1 ... 87.1 90.0
  * lon      (lon) float64 -180.0 -174.4 ... 174.4 180.0
Data variables:
    thetao   (time, depth, lat, lon) float64 ...  # Temperature
    so       (time, depth, lat, lon) float64 ...  # Salinity
    uo       (time, depth, lat, lon) float64 ...  # U velocity
    vo       (time, depth, lat, lon) float64 ...  # V velocity
```

**GLONET Format (Preprocessed)**
```
<xarray.Dataset>
Dimensions:  (time: 30, channel: 40, lat: 64, lon: 64)
Coordinates:
  * time     (time) datetime64[ns] 2023-01-01 ... 2023-01-30
  * channel  (channel) int 1 2 ... 40
  * lat      (lat) float64 -90.0 -87.1 ... 87.1 90.0
  * lon      (lon) float64 -180.0 -174.4 ... 174.4 180.0
Data variables:
    data     (time, channel, lat, lon) float64 ...
```

**Important Notes:**
- Time dimension must be continuous and evenly spaced
- Spatial dimensions must be on a regular grid
- Missing values should be marked as NaN
- Minimum of `sequence_length + forecast_horizon` time steps required

#### Supported Data Sources

- **Copernicus Marine Service**: GLORYS12V1 reanalysis (recommended)
- **Custom NetCDF**: Any CF-compliant NetCDF file
- **Multiple files**: Can load variables from separate files

</br></br>





## Environment Setup

### 1. Install Conda/Mamba

If you don't have conda installed:

```bash
# Install Miniforge (includes mamba)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Or use existing conda and install mamba:
```bash
conda install -c conda-forge mamba
```

### 2. Create Environment from YAML

Navigate to the training directory and create the environment:

```bash
cd /path/to/glonet/training

# Using mamba (faster)
mamba env create -f environment.yml

# Or using conda
conda env create -f environment.yml
```

### 3. Activate Environment

```bash
conda activate glon
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9+

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Hydra
python -c "import hydra; print(f'Hydra: {hydra.__version__}')"

# Check xarray
python -c "import xarray; print(f'xarray: {xarray.__version__}')"
```

</br></br>

## Prepare Training Dataset

### Download GLORYS12 from Copernicus Marine

A download script is provided to automatically fetch and preprocess GLORYS12 data from Copernicus Marine Service.

**Default Script Location:** `../glonet_daily_forecast_local/down_glorys12_training.py`


**What the script does:**
- Downloads **Global Ocean Physics Reanalysis** data at 21 depth levels (surface + upper ocean + deep ocean)
- Regrids from native 1/12° resolution to 1/4° resolution using bilinear interpolation
- Concatenates all variables (temperature, salinity, velocities, SSH) into channel dimension
- Outputs a single NetCDF file in GLONET-ready format

### Prerequisites for Download

1. **Copernicus Marine Account**
   - Create a free account at: https://marine.copernicus.eu/
   - Accept the GLORYS12 dataset license terms

2. **Configure Credentials**
   ```bash
   # Set up Copernicus Marine credentials (one-time setup)
   copernicusmarine login
   # Enter your username and password when prompted
   ```

3. **Required Packages** (included in environment.yml)
   - `copernicusmarine` - Data download client
   - `xesmf` - Regridding library
   - `xarray`, `numpy` - Data processing

### Using the Download Script

Navigate to the download script directory:

```bash
cd /path/to/glonet/glonet_daily_forecast_local/
```

#### Basic Usage

```bash
python down_glorys12_training.py \
    --start_date "1993-01-01" \
    --end_date "2025-12-31"
```

This downloads data from 1993 January 1 to 2025 December 31, and saves to:
```
/Odyssey/public/glonet/glorys12_2020-01-01_to_2020-12-31_init_states/combined_input.nc
```
*If you lanch this script outside of IMTA server, please specify output file path.*

#### Specify Custom Output Path

```bash
python down_glorys12_training.py \
    --start_date "2020-01-01" \
    --end_date "2020-12-31" \
    --out_path "/data/glonet/training_data/"
```

#### Command Line Arguments

| Argument | Short | Required | Description | Example |
|----------|-------|----------|-------------|---------|
| `--start_date` | `-s` | Yes | Start date in YYYY-MM-DD format | `"2020-01-01"` |
| `--end_date` | `-e` | Yes | End date in YYYY-MM-DD format | `"2020-12-31"` |
| `--out_path` | `-o` | No (but Yes for non-IMTA) | Custom output directory | `"/data/glonet"` |

### What Gets Downloaded

The script downloads **GLORYS12 daily data at 21 depth levels**, organized in 3 groups:

**Group 1 - Surface (1 level @ 0.5m depth)**
- Variables: SSH (zos), temperature (thetao), salinity (so), U velocity (uo), V velocity (vo)
- **5 channels**

**Group 2 - Upper Ocean (10 levels)**
- Depths: 50, 100, 150, 222, 318, 380, 450, 540, 640, 763m
- Variables: temperature, salinity, U velocity, V velocity
- **40 channels** (4 variables × 10 depths)

**Group 3 - Deep Ocean (10 levels)**
- Depths: 902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274m
- Variables: temperature, salinity, U velocity, V velocity
- **40 channels** (4 variables × 10 depths)

**Total Output: 85 channels**

### Output Data Format

The downloaded file has the GLONET-ready format:

```
<xarray.Dataset>
Dimensions:  (time: N, ch: 85, lat: 680, lon: 1440)
Coordinates:
  * time     (time) datetime64[ns] 2020-01-01 ... 2020-12-31
  * ch       (ch) int 0 1 2 ... 84
  * lat      (lat) float64 -80.0 -79.75 ... 89.75 90.0
  * lon      (lon) float64 -180.0 -179.75 ... 179.75 180.0
Data variables:
    data     (time, ch, lat, lon) float64 ...
```

**Spatial Coverage:**
- Latitude: -80° to 90° (near-global, excluding Antarctic shelves)
- Longitude: -180° to 180° (global)
- Resolution: Re-interpolated 1/4° (~25km at equator)

### Download Examples

#### Single Year for Training
```bash
python down_glorys12_training.py \
    -s "2020-01-01" \
    -e "2020-12-31" \
    -o ~/data/glonet/train_2020/
```

#### Multiple Years (Sequential Downloads)
```bash
# Download 2018
python down_glorys12_training.py -s "2018-01-01" -e "2018-12-31" -o ~/data/glonet/2018/

# Download 2019
python down_glorys12_training.py -s "2019-01-01" -e "2019-12-31" -o ~/data/glonet/2019/

# Download 2020
python down_glorys12_training.py -s "2020-01-01" -e "2020-12-31" -o ~/data/glonet/2020/
```

#### Short Period for Testing
```bash
# Download just one month for quick testing
python down_glorys12_training.py \
    -s "2020-01-01" \
    -e "2020-01-31" \
    -o ~/data/glonet/test/
```

### Download Time & Storage Estimates

**Download Times** (varies with network speed):
| Period | Approximate Time |
|--------|------------------|
| 1 month | 15-30 minutes |
| 3 months | 1-2 hours |
| 1 year | 6-8 hours |
| 5 years | 30-40 hours |

**Storage Requirements:**
| Period | File Size |
|--------|-----------|
| 1 month | ~10-15 GB |
| 3 months | ~30-45 GB |
| 1 year | ~120-180 GB |
| 5 years | ~600-900 GB |

💡 **Tip**: For training, download at least 1 year of data for robust model learning.

### Troubleshooting Downloads

#### Authentication Issues
```bash
# Re-configure credentials
copernicusmarine login

# Or set environment variables
export COPERNICUS_MARINE_SERVICE_USERNAME="your_username"
export COPERNICUS_MARINE_SERVICE_PASSWORD="your_password"
```

#### Temporal availability
|Availiable data period | 01/01/1993 ~ 23/12/2025 |
| - | - |
at current time : 01/02/2026


#### Network Timeout or Interruption
The script downloads data sequentially. If interrupted:
1. Note the last successful date from console output
2. Resume from that date with a new command
```bash
python down_glorys12_training.py -s "2020-06-15" -e "2020-12-31" -o same_output_directory/
```

#### **[Important]**  Regridding Weights
The script requires pre-computed regridding weights at:
```
/Odyssey/public/glonet/TrainedWeights/xe_weights14/L*.nc
```
If these files are missing, contact your system administrator or generate them using `xesmf.Regridder`.

### Using Downloaded Data for Training

Once the download is complete, use the data in your training:

```bash
cd ../training/

python train.py \
    data.input_paths="/Odyssey/public/glonet/glorys12_2020-01-01_to_2020-12-31_init_states/combined_input.nc" \
    data.forecast_horizon=7
```

Or with multiple years (concatenate files manually using xarray if needed):
```bash
python train.py \
    data.input_paths="/data/glonet/2018/combined_input.nc" \
    training.epochs=100 \
    training.batch_size=4
```

</br></br>






## Running Training

### 1. Configuration Structure

The training system uses Hydra for configuration management:
to intuitive module handling, the training uses data, model and training configurations. </br>

* Data: configuration about training dataset. </br>
* Model: GLONET model structure. </br>
* Training: training configuration.

```
config/
├── config.yaml              # Main configuration
├── data/
│   └── GLORYS12_0125.yaml  # Dataset configuration
├── model/
│   ├── glonet.yaml         # Standard model
│   └── glonet_small.yaml   # Small model for testing
└── training/
    ├── default.yaml        # Default training settings
    └── fast.yaml           # Fast training for testing
```

### 2. Basic Training

#### Quick Start (Default Configuration)

```bash
# Train with default settings
python train.py
```

This will:
- Load default configurations from `config/config.yaml`
- Create a GLONET model
- Train on the specified dataset
- Save checkpoints and logs to `outputs/YYYY-MM-DD_HH-MM-SS/`

#### Specify Data Path

```bash
# Train with your NetCDF data
python train.py data.input_paths="/path/to/your/ocean_data.nc"
```

#### Quick Test Run

```bash
# Fast test with small model (useful for debugging)
python train.py model=glonet_small training=fast training.epochs=2
```s

### 3. Configuration Override Examples

Hydra allows you to override any configuration parameter from the command line:

#### Data Configuration

```bash
# Specify data file
python train.py data.input_paths="data/glorys12_1994.nc"



# Change forecast horizon (prediction time steps)
python train.py data.forecast_horizon=7

# Adjust data splits
python train.py data.train_split=0.7 data.val_split=0.15 data.test_split=0.15

# Enable preprocessing
python train.py data.preprocessing.normalize=true data.preprocessing.crop_size=[128,128]
```

#### Model Configuration

**Before starting** : Hyper parameter `patching` and `cropping` is not available in this phase.
Changing these parameters will destroy the original GLONET learning structure.

Model Configuration can be modified by changing model parameters for further GLONET studies (e.g. structure dev, dataset changes...) </br>
**Do not change to keep desgined GLONET fuctionality**

```bash
# Use small model
python train.py model=glonet_small

# Change model dimensions [T, C, H, W]
python train.py model.dim=[7,40,256,256]

# Adjust layer depths
python train.py model.NT=4 model.NS=16

# Change device
python train.py device=cuda:1  # Use second GPU
python train.py device=cpu     # Use CPU
```

If the glonet structure is modified, You could change GLONET parameters like above. 
And, you can change data config as well, for example :

```bash
# Select specific variables (Dataset changed)
python train.py data.variables="['thetao','so','uo','vo']"

# Change sequence length (GLONET input time steps changed)
python train.py data.sequence_length=3
```

#### Training Configuration

```bash
# Change epochs and batch size
python train.py training.epochs=100 training.batch_size=8

# Adjust learning rate
python train.py training.learning_rate=0.0001

# Enable mixed precision training (faster, less memory)
python train.py training.use_amp=true

# Gradient clipping
python train.py training.grad_clip_norm=1.0

# Change optimizer
python train.py training.optimizer.weight_decay=0.01

# Use different loss function
python train.py training.loss._target_=torch.nn.L1Loss
```

#### Combined Examples

```bash
# Full training run with custom settings
python train.py \
    data.input_paths="data/glorys12_full.nc" \
    data.sequence_length=14 \
    data.forecast_horizon=7 \
    model=glonet \
    training.epochs=200 \
    training.batch_size=4 \
    training.learning_rate=0.0005 \
    training.use_amp=true \
    device=cuda:0

# Quick debug run
python train.py \
    model=glonet \
    training=fast \
    training.epochs=1 \
    training.batch_size=2 \
    data.train_split=0.1
```

### 4. Multiple Data Files

You can load variables from different NetCDF files:

```bash
python train.py data.input_paths="{'thetao':'temp.nc','so':'sal.nc','uo':'u_vel.nc','vo':'v_vel.nc'}"
```

### 5. Hyperparameter Sweeps

Run multiple experiments with different hyperparameters:

```bash
# Sweep over learning rates (creates multiple runs)
python train.py -m training.learning_rate=0.0001,0.0005,0.001

# Sweep over multiple parameters
python train.py -m training.batch_size=4,8 training.learning_rate=0.0001,0.001

# Sweep over seeds for multiple trials
python train.py -m seed=42,123,456,789,1024
```

</br></br>




## Training Monitor

### 1. TensorBoard Setup

TensorBoard is automatically integrated into the training pipeline for real-time monitoring.

#### Starting TensorBoard

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=outputs/

# Or specify a specific run
tensorboard --logdir=outputs/2024-01-15_10-30-45/

# Custom port
tensorboard --logdir=outputs/ --port=6007

# Allow external connections (on remote server)
tensorboard --logdir=outputs/ --bind_all
```

#### Access TensorBoard

Open your browser and navigate to:
- Local: `http://localhost:6006`
- Remote server: `http://<server-ip>:6006`

### 2. Monitored Metrics

TensorBoard logs the following metrics:

#### Training Metrics
- **Loss curves**: Train/validation/test loss per epoch
- **Learning rate**: LR schedule over time
- **Gradient norms**: Monitor gradient flow
- **Loss components**: Individual loss terms (if using composite loss)

<!-- #### Model Metrics
- **Parameter histograms**: Weight distributions over time
- **Activation statistics**: Layer activation patterns
- **Gradient histograms**: Gradient distributions

#### Performance Metrics
- **Training speed**: Samples/second, epoch time
- **GPU utilization**: Memory usage, compute utilization
- **Batch processing time**: Forward/backward pass timing

#### Forecast Quality
- **MSE/MAE**: Mean squared/absolute error
- **RMSE**: Root mean squared error
- **Correlation**: Spatial correlation with ground truth
- **Anomaly detection**: Error distribution analysis -->

### 3. TensorBoard Features

#### Scalars
View loss curves and metrics:
```
Scalars tab → Select metrics to compare
- loss/train
- loss/validation
- loss/test
- metrics/rmse
- metrics/mae
```

#### Hyperparameters
Compare runs with different hyperparameters:
```
HParams tab → Compare metrics across runs
```

### 4. Monitoring During Training

**Best Practices:**

1. **Check TensorBoard regularly** during long training runs
2. **Monitor validation loss** for overfitting
3. **Watch gradient norms** for vanishing/exploding gradients
4. **Compare multiple runs** to find best hyperparameters
5. **Use tags** to organize experiments

**Warning Signs:**
- Loss not decreasing after several epochs
- Validation loss increasing while training loss decreases (overfitting)
- Gradient norms exploding (>10) or vanishing (<0.001)
- NaN losses (reduce learning rate or check data)

### 5. Logging Custom Metrics

To add custom metrics to TensorBoard (for advanced users):

```python
# In train.py or glonetLit.py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='outputs/your_run/')

# Log scalar
writer.add_scalar('custom/metric', value, epoch)

# Log image
writer.add_image('predictions/sample', image_tensor, epoch)

# Log histogram
writer.add_histogram('weights/layer1', weight_tensor, epoch)
```
</br></br>




## Advanced Usage

### 1. Distributed Training

For multi-GPU training:

```bash
# PyTorch DDP (recommended)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    training.distributed=true

# Or use accelerate (if integrated)
accelerate launch train.py
```

### 2. Resume Training (Fine-tunning)

```bash
# Resume from checkpoint
python train.py training.resume_from="outputs/2024-01-15/checkpoints/best_model.pt"
```

### 3. Evaluation Only

```bash
# Evaluate a trained model
python train.py \
    mode=eval \
    training.checkpoint_path="outputs/2024-01-15/checkpoints/best_model.pt" \
    data.input_paths="data/test_data.nc"
```

### 4. Export Configuration

```bash
# Save current configuration to file
python train.py --cfg job > my_config.yaml

# Use custom config file
python train.py --config-name=my_config
```

### 5. Output Organization

Outputs are automatically organized:

```
outputs/
├── 2024-01-15_10-30-45/        # Run timestamp
│   ├── .hydra/                  # Hydra configs
│   │   ├── config.yaml         # Used configuration
│   │   └── overrides.yaml      # CLI overrides
│   ├── checkpoints/             # Model checkpoints
│   │   ├── best_model.pt       # Best validation loss
│   │   ├── last_model.pt       # Latest checkpoint
│   │   └── epoch_*.pt          # Periodic checkpoints
│   ├── logs/                    # TensorBoard logs
│   │   └── events.out.tfevents.*
│   └── train.log                # Console output log
└── multirun/                    # Multi-run sweeps
    └── 2024-01-15_11-00-00/
        ├── 0/                   # First config
        ├── 1/                   # Second config
        └── ...
```

### Getting Help

1. **Check logs**: `outputs/YYYY-MM-DD_HH-MM-SS/train.log`
2. **Validate data**: Use xarray to inspect your NetCDF files
3. **Test with dummy data**: Use small synthetic dataset first
4. **Check TensorBoard**: Look for anomalies in training curves
5. **Review configuration**: Ensure all paths and parameters are correct

### System Requirements Check

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check disk space
df -h

# Check Python packages
pip list | grep -E "torch|hydra|xarray|tensorboard"
```

</br></br>




## Additional Resources

- **DATASET_README.md**: Detailed dataset integration guide
- **HYDRA_README.md**: Advanced Hydra configuration examples
- **Hydra Documentation**: https://hydra.cc/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorBoard Documentation**: https://www.tensorflow.org/tensorboard

</br></br>





## Quick Reference

### Essential Commands

```bash
# Setup
conda activate glonet

# Basic training
python train.py

# Training with custom data
python train.py data.input_paths="your_data.nc"

# Monitor training
tensorboard --logdir=outputs/

# Quick test
python train.py model=glonet_small training=fast training.epochs=1

# Sweep hyperparameters
python train.py -m training.learning_rate=0.0001,0.001 training.batch_size=4,8
```
</br></br>




## License

Mercator Ocean International

</br></br>




## Contact

* [Jungwon LEE](jungwon.lee@imt-atlantique.fr) [ jungwon.lee@imt-atlantique.fr ]
* [Anass El Aouni](aelaouni@mercator-ocean.fr) [ aelaouni@mercator-ocean.fr ]