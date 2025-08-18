#!/usr/bin/env python3
"""
GLONET Training Script with Hydra Configuration
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.utils.data
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from glonet import Glonet
from blocks import *
from NN import *
from NO import *
from dataset import XrDataset, create_xr_datasets

# Set up logging
log = logging.getLogger(__name__)


class GlonetTrainer:
    """Training class for GLONET model"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Initialize loss function
        self.criterion = instantiate(cfg.training.loss)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Model initialized with {self.count_parameters()} parameters")
        log.info(f"Using device: {self.device}")
    
    
    def _build_model(self) -> nn.Module:
        """Build the GLONET model"""
        
        model_cfg = self.cfg.model
        
        # Extract parameters from config
        # dim = model_cfg.dim
        # dT = model_cfg.get('dT', 64)
        # dS = model_cfg.get('dS', 64)
        # NT = model_cfg.get('NT', 2)
        # NS = model_cfg.get('NS', 8)
        # ker = model_cfg.get('ker', [3, 5, 7])
        # groups = model_cfg.get('groups', 8)
        
        # model = Glonet(
        #     dim=dim,
        #     dT=dT,
        #     dS=dS,
        #     NT=NT,
        #     NS=NS,
        #     ker=ker,
        #     groups=groups,
        #     device=self.device.type
        # )
        
        return instantiate(model_cfg).to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer"""
        
        return instantiate(
            self.cfg.training.optimizer,
            params=self.model.parameters()
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        
        if 'scheduler' in self.cfg.training:
            return instantiate(
                self.cfg.training.scheduler,
                optimizer=self.optimizer
            )
        return None
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, 
                    train_loader : torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if hasattr(self.cfg.training, 'grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.training.grad_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.cfg.log_interval == 0:
                log.info(f'Train Epoch: {self.current_epoch} [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.6f}')
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self, 
                 val_loader : torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    
    def save_checkpoint(self, 
                        epoch : int, 
                        is_best : bool = False) -> None :
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_yaml(self.cfg)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.ckpt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.ckpt"
            torch.save(checkpoint, best_path)
            log.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
    
    
    def train(self, 
              train_loader : torch.utils.data.DataLoader, 
              val_loader : torch.utils.data.DataLoader) -> float:
        """Main training loop"""
        
        log.info("Starting training...")
        
        for epoch in range(1, self.cfg.training.epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Check if this is the best model
            current_val_loss = val_metrics['val_loss']
            is_bestmodel = current_val_loss < self.best_val_loss
            if is_bestmodel:
                self.best_val_loss = current_val_loss
            
            # Log metrics
            log.info(f"Epoch {epoch}/{self.cfg.training.epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.6f}, "
                    f"Val Loss: {val_metrics['val_loss']:.6f}")
            
            # Save checkpoint
            if self.cfg.save_checkpoint and epoch % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(epoch, is_bestmodel)
        
        log.info("Training completed!")
        
        return self.best_val_loss


def create_data_loaders(cfg) -> tuple:
    """Create data loaders using XrDataset"""
    
    try:
        # Create datasets using the new XrDataset class
        train_dataset, val_dataset, test_dataset = create_xr_datasets(cfg)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False
        )
        
        log.info(f"Created data loaders - Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        log.error(f"Failed to create data loaders: {e}")
        sys.exit(1)
        
        # log.warning(f"Failed to create XrDataset: {e}")
        # log.info("Falling back to dummy data loader...")
        
        # Fallback to dummy data if XrDataset fails
        # train_loader = create_dummy_data_loader(cfg, 'train')
        # val_loader = create_dummy_data_loader(cfg, 'val')
        # # test_loader = create_dummy_data_loader(cfg, 'test')
        
        # return train_loader, val_loader, test_loader


# def create_dummy_data_loader(cfg: DictConfig, split: str = 'train'):
#     """Create a dummy data loader for demonstration"""
#     # This is a placeholder - replace with your actual data loading logic
#     import torch.utils.data as data
    
#     batch_size = cfg.training.batch_size
#     dim = cfg.model.dim
    
#     # Create dummy dataset
#     class DummyDataset(data.Dataset):
#         def __init__(self, size=1000):
#             self.size = size
            
#         def __len__(self):
#             return self.size
            
#         def __getitem__(self, idx):
#             # Input shape: [T, C, H, W]
#             input_data = torch.randn(*dim)
#             # Target shape: [C, H, W] (forecast for first timestep)
#             target_data = torch.randn(dim[1], dim[2], dim[3])
#             return input_data, target_data
    
#     dataset = DummyDataset(size=800 if split == 'train' else 200)
    
#     return data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == 'train'),
#         num_workers=cfg.data.num_workers,
#         pin_memory=cfg.data.pin_memory
#     )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> float:
    """Main training function"""
    
    # Print configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Create trainer
    trainer = GlonetTrainer(cfg)
    
    # Create data loaders
    log.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(cfg)
    
    # Train the model
    best_val_loss = trainer.train(train_loader, val_loader)
    
    log.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    return best_val_loss


if __name__ == "__main__":
    main()
