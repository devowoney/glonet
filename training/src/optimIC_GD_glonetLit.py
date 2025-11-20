import torch
from torch import nn
import torch.nn.functional as F
# import torch.fft
import numpy as np
import torch.optim as optimizer
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import xarray as xr
from hydra.utils import instantiate

# from blocks import *
# from NN import *

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src/glonet"))
from modelp2 import Glonet 

import logging
log = logging.getLogger(__name__)

import torch.utils.checkpoint as checkpoint_util




class GlonetGradientCheckpointing(Glonet) :
    def __init__(self, shape_in, hid_S=256, hid_T=128, N_S=2, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super().__init__(shape_in, hid_S, hid_T, N_S, N_T, incep_ker, groups)
    
    def forward(self, input_st_tensors):
        B, T, C, H, W = input_st_tensors.shape
        
        # Use gradient checkpointing for memory-intensive operations
        def compute_spatial_features(x):
            skip_feature = self.jump(x)
            spatial_feature = self.space(x)
            return skip_feature, spatial_feature
        
        def compute_latent_features(spatial_feature):
            spatial_feature = spatial_feature.reshape(-1, C, H, W)
            return self.maps(spatial_feature)
        
        def compute_temporal_features(spatial_embed):
            return self.dynamics(spatial_embed)
        
        def compute_predictions(spatialtemporal_embed, spatial_skip_feature):
            return self.mapsback(spatialtemporal_embed, spatial_skip_feature)
        
        # Apply gradient checkpointing to reduce memory usage
        skip_feature, spatial_feature = checkpoint_util.checkpoint(
            compute_spatial_features, input_st_tensors, use_reentrant=False
        )
        
        spatial_embed, spatial_skip_feature = checkpoint_util.checkpoint(
            compute_latent_features, spatial_feature, use_reentrant=False
        )
        
        # Reshape for temporal processing
        _, C_, H_, W_ = spatial_embed.shape
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_)
        
        spatialtemporal_embed = checkpoint_util.checkpoint(
            compute_temporal_features, spatial_embed, use_reentrant=False
        )
        
        # Reshape back
        spatialtemporal_embed = spatialtemporal_embed.reshape(B*T, C_, H_, W_)
        
        predictions = checkpoint_util.checkpoint(
            compute_predictions, spatialtemporal_embed, spatial_skip_feature, use_reentrant=False
        )
        
        # Final computation
        predictions = 0.05 * predictions.reshape(B, T, C, H, W) + skip_feature
        
        return predictions



class GlonetGradientInitialCondition(pl.LightningModule) :
    
    def __init__(self, 
                 cfg = None) :
        super().__init__()
        
        # Store config for optimizer parameters
        self.cfg = cfg
        
        # init_input will be created once 
        self.init_sequence = None
        self.target = None
        self.loss_fn = None
        self._ic_optimizer = None
        self._ic_scheduler = None
        
        # initialize mean and std for standarization
        self.dataset = None
        self.mean = None
        self.std = None
        
        self.save_path = None
        self.current_loss = float('inf')
        self.best_loss = float('inf')
        self.best_init = None
        
        # Load pytorch checkpoint
        log.info("Loading checkpoint...")
        self.checkpoint = torch.load(self.cfg.model_path, map_location=torch.device('cuda'))

        # Create new gradient checkpointing model instance
        self.gradcheckp_model = GlonetGradientCheckpointing(shape_in=(2, 85, 672, 1440))

        # Load the same weights from the original model
        self.gradcheckp_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.gradcheckp_model.eval()
        
        # Freeze model parameters — we only optimize the initial condition
        for param in self.gradcheckp_model.parameters():
            param.requires_grad = False
            
        log.info(f"Frozen {sum(1 for _ in self.gradcheckp_model.parameters())} parameters")

    def forward(self, 
                x : torch.Tensor = None) -> torch.Tensor :
        """Forward pass through the saved model with forecast window iterations."""
        
        if x is not None and not x.requires_grad:
            log.warning(f"Input tensor does not require grad: {x.requires_grad}")
        
        try:
            # !!! Not explicit device handling : model is hardcoded to cuda in JIT
            with torch.enable_grad():
                return self.gradcheckp_model(x)
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"[Error]: CUDA device issue in JIT model: {e}")
                print(f"[Error]: Your JIT model was saved with hardcoded CUDA devices.")
                print(f"[Error]: You need to retrain/resave the model without hardcoded devices,")
                print(f"[Error]: or run on a machine with CUDA available.")
                raise RuntimeError("JIT model has hardcoded CUDA device but CUDA is not available") from e
            else:
                raise e

    def step(self) -> float :
        """Single optimization step."""
        
        y_hat = self.forward(self.init_input)
        loss = self.loss_fn(y_hat, self.target)
        
        return loss

    def training_step(self) -> float :
        """Define training step - init_input is already created in on_train_start."""
        
        # For JIT models, we might need to explicitly enable gradients
        with torch.enable_grad():
            train_loss = self.step()

            # Log metrics
            self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
            
            return train_loss

    def on_train_start(self, 
                       batch : tuple) -> None :
        """Initialize init_input and optimizer once before training starts."""
        
        # Create Parameter for optimization - keep this standardized so gradients flow
        x, y = batch
        self.init_input = nn.Parameter(x.detach().clone().unsqueeze(0), requires_grad=True)
        self.target = y
        
        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler"""
    
        optimizer = instantiate(self.cfg.training.optimizer, 
                                params=self.parameters())
        
        # Check if scheduler is configured
        if 'scheduler' in self.cfg.training:
            scheduler = instantiate(self.cfg.training.scheduler, 
                                    optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        return optimizer
    


