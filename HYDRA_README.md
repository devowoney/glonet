# GLONET with Hydra Configuration

This document explains how to use Hydra configuration management with your GLONET ocean forecasting model.

## Installation

First, install the required packages:

```bash
pip install hydra-core omegaconf
```

## Configuration Structure

The configuration system is organized as follows:

```
config/
├── config.yaml           # Main configuration file
├── model/                 # Model configurations
│   ├── glonet.yaml       # Standard GLONET configuration
│   └── glonet_small.yaml # Smaller model for testing
├── training/              # Training configurations
│   ├── default.yaml      # Default training settings
│   └── fast.yaml         # Fast training for testing
├── data/                  # Data configurations
│   ├── default.yaml      # Default data settings
│   └── ocean.yaml        # Ocean-specific data settings
└── experiment/            # Pre-defined experiments
    ├── quick_test.yaml    # Quick test experiment
    └── ocean_full.yaml    # Full ocean forecasting experiment
```

## Basic Usage

### 1. Using the Example Script

Run the example script to test your setup:

```bash
python example.py
```

This will:
- Load the default configuration
- Create a GLONET model
- Test it with dummy data

### 2. Using Different Configurations

You can specify different configurations using Hydra's command line interface:

```bash
# Use small model for testing
python example.py model=glonet_small

# Use fast training settings
python example.py training=fast

# Use ocean-specific data configuration
python example.py data=ocean

# Combine multiple configurations
python example.py model=glonet_small training=fast data=ocean
```

### 3. Using Pre-defined Experiments

Run complete experiment configurations:

```bash
# Quick test with small model and fast training
python example.py --config-path=config/experiment --config-name=quick_test

# Full ocean forecasting experiment
python example.py --config-path=config/experiment --config-name=ocean_full
```

### 4. Overriding Configuration Parameters

Override any parameter from the command line:

```bash
# Change batch size and learning rate
python train.py training.batch_size=16 training.learning_rate=0.001

# Change model dimensions
python train.py model.dim=[3,5,256,256] model.dT=128

# Change device
python train.py device=cpu
```

## Training with Hydra

To start training with the full training script:

```bash
# Basic training with default settings
python train.py

# Training with specific experiment
python train.py --config-path=config/experiment --config-name=ocean_full

# Training with overrides
python train.py training.epochs=50 training.batch_size=8 model=glonet_small
```

## Configuration Management Utilities

Use the configuration utility script to manage your configurations:

```bash
# List all available configurations
python config_utils.py list

# Print a specific configuration
python config_utils.py print config/model/glonet.yaml

# Validate a configuration file
python config_utils.py validate config/config.yaml

# Create a new configuration from template
python config_utils.py create my_model --template model
```

## Configuration Files Explained

### Main Configuration (`config/config.yaml`)
- Defines default configurations to use
- Sets global parameters like project name, device, seed
- Configures Hydra output directory structure

### Model Configuration (`config/model/glonet.yaml`)
- Defines GLONET architecture parameters
- Includes dimensions, layer counts, kernel sizes
- Can be easily modified for different model variants

### Training Configuration (`config/training/default.yaml`)
- Sets training hyperparameters
- Defines optimizer and scheduler settings
- Includes early stopping and gradient clipping options

### Data Configuration (`config/data/default.yaml`)
- Defines data loading parameters
- Sets paths to input files
- Includes preprocessing and augmentation settings

## Creating Custom Configurations

### 1. Create a New Model Configuration

```yaml
# config/model/my_model.yaml
_target_: src.glonet.Glonet
dim: [3, 6, 256, 256]  # Larger model
dT: 128
dS: 128
NT: 3
NS: 12
ker: [3, 5, 7, 9]
groups: 16
```

### 2. Create a New Training Configuration

```yaml
# config/training/my_training.yaml
epochs: 300
batch_size: 4
learning_rate: 0.0001

optimizer:
  _target_: torch.optim.AdamW
  lr: ${training.learning_rate}
  weight_decay: 0.01
  betas: [0.9, 0.95]

scheduler:
  _target_: torch.optim.lr_scheduler.OneCycleLR
  max_lr: ${training.learning_rate}
  epochs: ${training.epochs}
  pct_start: 0.1

loss:
  _target_: torch.nn.L1Loss

use_amp: true
grad_clip_norm: 1.0
```

### 3. Create a New Experiment

```yaml
# config/experiment/my_experiment.yaml
# @package _global_
defaults:
  - /model: my_model
  - /training: my_training
  - /data: ocean

experiment_name: "my_custom_experiment"
seed: 123

# Override specific parameters
training:
  batch_size: 2  # Reduce for large model

output_dir: "outputs/my_experiment"
```

## Advanced Features

### 1. Hyperparameter Sweeps

You can easily run hyperparameter sweeps with Hydra:

```bash
# Sweep over learning rates
python train.py -m training.learning_rate=0.001,0.01,0.1

# Sweep over multiple parameters
python train.py -m training.batch_size=4,8,16 training.learning_rate=0.001,0.01
```

### 2. Multi-run with Different Seeds

```bash
# Run multiple times with different seeds
python train.py -m seed=42,123,456
```

### 3. Using Hydra Plugins

You can integrate with various Hydra plugins for:
- Job scheduling (Slurm, etc.)
- Hyperparameter optimization (Optuna, etc.)
- Logging (Weights & Biases, etc.)

## Output Organization

Hydra automatically organizes outputs in timestamped directories:

```
outputs/
├── 2024-01-15_10-30-45/   # Timestamped run directory
│   ├── .hydra/            # Hydra configuration logs
│   ├── train.log          # Training logs
│   └── checkpoints/       # Model checkpoints
└── multirun/              # Multi-run outputs
    └── 2024-01-15_10-35-12/
```

## Best Practices

1. **Use composition**: Break configurations into reusable components
2. **Name configurations clearly**: Use descriptive names for different variants
3. **Document experiments**: Use the experiment configs for reproducible research
4. **Version control**: Keep all configuration files in version control
5. **Test configurations**: Use the quick_test experiment to validate changes

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure hydra-core and omegaconf are installed
2. **CUDA errors**: Set `device: cpu` in config if CUDA is not available
3. **Memory errors**: Reduce batch size or model size in configuration
4. **Path errors**: Use absolute paths or ensure working directory is correct

### Getting Help

- Check the configuration with: `python config_utils.py validate <config_path>`
- Print configuration with: `python config_utils.py print <config_path>`
- Use `python example.py --help` to see all available options
