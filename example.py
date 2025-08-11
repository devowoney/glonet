#!/usr/bin/env python3
"""
Example script showing how to use GLONET with Hydra configuration
Run this after installing hydra-core and omegaconf
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Try to import hydra components
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    HYDRA_AVAILABLE = True
except ImportError:
    print("Hydra not available. Please install with: pip install hydra-core omegaconf")
    HYDRA_AVAILABLE = False

from glonet import Glonet


def load_config_manually(config_name: str = "config"):
    """Load configuration manually without hydra decorator"""
    if not HYDRA_AVAILABLE:
        print("Using manual configuration since Hydra is not available")
        # Return a manual configuration
        return {
            'model': {
                'dim': [2, 5, 128, 128],
                'dT': 64,
                'dS': 64, 
                'NT': 2,
                'NS': 8,
                'ker': [3, 5, 7],
                'groups': 8
            },
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'seed': 42
        }
    
    # Load with OmegaConf
    config_path = Path("config") / f"{config_name}.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return cfg
    else:
        print(f"Config file {config_path} not found")
        return None


def create_model_from_config(cfg):
    """Create GLONET model from configuration"""
    if HYDRA_AVAILABLE and hasattr(cfg, 'model'):
        model_cfg = cfg.model
    else:
        model_cfg = cfg.get('model', cfg)
    
    # Extract parameters
    dim = model_cfg.get('dim', [2, 5, 128, 128])
    dT = model_cfg.get('dT', 64)
    dS = model_cfg.get('dS', 64)
    NT = model_cfg.get('NT', 2)
    NS = model_cfg.get('NS', 8)
    ker = model_cfg.get('ker', [3, 5, 7])
    groups = model_cfg.get('groups', 8)
    
    print(f"Creating GLONET with parameters:")
    print(f"  dim: {dim}")
    print(f"  dT: {dT}, dS: {dS}")
    print(f"  NT: {NT}, NS: {NS}")
    print(f"  kernels: {ker}")
    print(f"  groups: {groups}")
    
    model = Glonet(
        dim=dim,
        dT=dT,
        dS=dS,
        NT=NT,
        NS=NS,
        ker=ker,
        groups=groups
    )
    
    return model


def test_model(model, device):
    """Test the model with dummy data"""
    model = model.to(device)
    model.eval()
    
    # Get model input dimensions
    # Assuming dim = [T, C, H, W]
    B = 1  # Batch size
    T, C, H, W = 2, 5, 128, 128  # Default dimensions
    
    print(f"Testing model with input shape: [{B}, {T}, {C}, {H}, {W}]")
    
    # Create dummy input
    with torch.no_grad():
        dummy_input = torch.randn(B, T, C, H, W).to(device)
        
        try:
            output = model(dummy_input)
            print(f"Model output shape: {output.shape}")
            print("✓ Model test successful!")
            return True
        except Exception as e:
            print(f"✗ Model test failed: {e}")
            return False


@hydra.main(version_base=None, config_path="config", config_name="config")
def main_with_hydra(cfg: DictConfig) -> None:
    """Main function using Hydra decorator (requires hydra-core)"""
    print("Using Hydra configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model_from_config(cfg)
    
    # Test model
    test_model(model, device)


def main_manual():
    """Main function without Hydra (fallback)"""
    print("Running without Hydra...")
    
    # Load configuration manually
    cfg = load_config_manually()
    
    # Set device
    device = torch.device(cfg.get('device', 'cpu'))
    print(f"Using device: {device}")
    
    # Create model
    model = create_model_from_config(cfg)
    
    # Test model
    test_model(model, device)


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        try:
            main_with_hydra()
        except Exception as e:
            print(f"Hydra main failed: {e}")
            print("Falling back to manual configuration...")
            main_manual()
    else:
        main_manual()
