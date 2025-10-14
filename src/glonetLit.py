import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from blocks import *
from NN import *

class Glonet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # Store config
        self.cfg = cfg
        
        # Model parameters from config
        self.model_cfg = cfg.model
        self.dim = self.model_cfg.dim
        T, C, H, W = self.dim
        self.Hn = int(H / 2 ** (self.model_cfg.NT / 2))
        self.Wn = int(W / 2 ** (self.model_cfg.NT / 2))
        
        # Initialize sub-modules
        self.space = mspace(T*self.model_cfg.dT, self.model_cfg.dS, self.model_cfg.NS, 
                            self.Hn, self.Wn, self.model_cfg.ker, self.model_cfg.groups)
        self.dynamics = tmp(dim=self.dim, n_heads=4, patch_size=[16,16])
        self.maps = Encoder(C, self.model_cfg.dT, self.model_cfg.NT)
        self.mapsback = Decoder(self.model_cfg.dT, C, self.model_cfg.NT)
        
        # Training parameters from config
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        
        # Loss function
        self.loss_fn = hydra.utils.instantiate(cfg.training.loss)
        
        # Save hyperparameters
        self.save_hyperparameters(cfg)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        # x = x.to('cuda:0')
        s_f = self.dynamics(x)
        s_e, s_k_f = self.maps(s_f.view(-1, C, H, W))
        t_e = self.space(s_e.view(B, T, *s_e.shape[1:]))
        forecast = self.mapsback(t_e.view(B*T, *t_e.shape[2:]), s_k_f)
        return forecast.view(B, T, C, H, W)[:, 0]
    
    def step(self, batch) :
        x, y = batch  # Assuming batch contains (input, target) pairs
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        train_loss = self.step(batch)

        # Log metrics
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        val_loss = self.step(batch)

        # Log metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning"""
        test_loss = self.step(batch)

        # Log metrics
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)

        return test_loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Instantiate optimizer from config
        optimizer = hydra.utils.instantiate(
            self.cfg.training.optimizer,
            params=self.parameters()
        )
        
        # Check if scheduler is configured
        if 'scheduler' in self.cfg.training:
            scheduler = hydra.utils.instantiate(
                self.cfg.training.scheduler,
                optimizer=optimizer
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
