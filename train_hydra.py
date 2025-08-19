"""
Hydra-powered training script for Glonet PyTorch Lightning model
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path


def create_dummy_data(cfg: DictConfig):
    """Create dummy data based on config parameters"""
    dim = cfg.model.dim
    batch_size = cfg.training.batch_size
    
    T, C, H, W = dim
    
    # Create larger dataset for proper training/validation split
    num_samples = 100
    
    # Input: (num_samples, T, C, H, W)
    inputs = torch.randn(num_samples, T, C, H, W)
    # Target: (num_samples, C, H, W) - forecast for next timestep
    targets = torch.randn(num_samples, C, H, W)
    
    # Split into train/val
    split_idx = int(0.8 * num_samples)
    
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    return (train_inputs, train_targets), (val_inputs, val_targets)

def create_dataloaders(cfg: DictConfig):
    """Create train and validation dataloaders"""
    (train_inputs, train_targets), (val_inputs, val_targets) = create_dummy_data(cfg)
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

def setup_callbacks(cfg: DictConfig):
    """Setup PyTorch Lightning callbacks"""
    callbacks = []
    
    # Model checkpoint callback
    if cfg.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.training.callbacks.checkpoint.get('monitor', 'val_loss'),
            dirpath=os.path.join(cfg.output_dir, 'checkpoints'),
            filename='glonet-{epoch:02d}-{val_loss:.2f}',
            save_top_k=cfg.training.callbacks.checkpoint.get('save_top_k', 3),
            mode=cfg.training.callbacks.checkpoint.get('mode', 'min'),
            save_last=True,
            every_n_epochs=cfg.get('checkpoint_interval', 1)
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.callbacks.early_stopping.get('monitor', 'val_loss'),
            min_delta=cfg.training.callbacks.early_stopping.get('min_delta', 0.001),
            patience=cfg.training.callbacks.early_stopping.get('patience', 10),
            verbose=True,
            mode=cfg.training.callbacks.early_stopping.get('mode', 'min')
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks

def setup_logger(cfg: DictConfig):
    """Setup PyTorch Lightning logger"""
    logger = TensorBoardLogger(
        save_dir=cfg.output_dir,
        name=cfg.project_name,
        version=cfg.experiment_name
    )
    return logger

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Create model
    model = hydra.utils.instantiate(cfg.model, cfg=cfg)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(cfg)
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs = cfg.training.epochs,
        accelerator = cfg.training.trainer.get('accelerator', 'auto'),
        devices = cfg.training.trainer.get('devices', 'auto'),
        logger = logger,
        callbacks = callbacks,
        log_every_n_steps = cfg.get('log_interval', 50),
        gradient_clip_val = cfg.training.get('grad_clip_norm', None),
        precision = cfg.training.trainer.get('precision', 16),
        deterministic = cfg.training.trainer.get('deterministic', True),
        enable_progress_bar = True
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, val_loader)
    
    # Save final model
    if cfg.save_checkpoint:
        final_model_path = os.path.join(cfg.output_dir, 'final_model.ckpt')
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
