#!/usr/bin/env python3
"""
Utility script for working with GLONET Hydra configurations
"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf


def print_config(config_path: str):
    """Print a configuration file"""
    cfg = OmegaConf.load(config_path)
    print(f"Configuration from {config_path}:")
    print(OmegaConf.to_yaml(cfg))


def validate_config(config_path: str):
    """Validate a configuration file"""
    try:
        cfg = OmegaConf.load(config_path)
        print(f"✓ Configuration {config_path} is valid")
        return True
    except Exception as e:
        print(f"✗ Configuration {config_path} is invalid: {e}")
        return False


def list_configs():
    """List all available configurations"""
    config_dir = Path("config")
    
    print("Available configurations:")
    print("\nModels:")
    for f in (config_dir / "model").glob("*.yaml"):
        print(f"  - {f.stem}")
    
    print("\nTraining:")
    for f in (config_dir / "training").glob("*.yaml"):
        print(f"  - {f.stem}")
    
    print("\nData:")
    for f in (config_dir / "data").glob("*.yaml"):
        print(f"  - {f.stem}")
    
    print("\nExperiments:")
    for f in (config_dir / "experiment").glob("*.yaml"):
        print(f"  - {f.stem}")


def create_custom_config(name: str, template: str = "default"):
    """Create a new configuration from template"""
    config_dir = Path("config")
    
    if template == "model":
        template_path = config_dir / "model" / "glonet.yaml"
        new_path = config_dir / "model" / f"{name}.yaml"
    elif template == "training":
        template_path = config_dir / "training" / "default.yaml"
        new_path = config_dir / "training" / f"{name}.yaml"
    elif template == "data":
        template_path = config_dir / "data" / "default.yaml"
        new_path = config_dir / "data" / f"{name}.yaml"
    elif template == "experiment":
        template_path = config_dir / "experiment" / "quick_test.yaml"
        new_path = config_dir / "experiment" / f"{name}.yaml"
    else:
        print(f"Unknown template type: {template}")
        return
    
    if not template_path.exists():
        print(f"Template {template_path} does not exist")
        return
    
    if new_path.exists():
        print(f"Configuration {new_path} already exists")
        return
    
    # Copy template
    cfg = OmegaConf.load(template_path)
    with open(new_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"Created new configuration: {new_path}")


def main():
    parser = argparse.ArgumentParser(description="GLONET Configuration Utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    subparsers.add_parser("list", help="List all available configurations")
    
    # Print command
    print_parser = subparsers.add_parser("print", help="Print a configuration file")
    print_parser.add_argument("config_path", help="Path to configuration file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config_path", help="Path to configuration file")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration from template")
    create_parser.add_argument("name", help="Name for the new configuration")
    create_parser.add_argument("--template", choices=["model", "training", "data", "experiment"], 
                              default="model", help="Template type")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_configs()
    elif args.command == "print":
        print_config(args.config_path)
    elif args.command == "validate":
        validate_config(args.config_path)
    elif args.command == "create":
        create_custom_config(args.name, args.template)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
