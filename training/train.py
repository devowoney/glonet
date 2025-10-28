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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
from dataset import GlonetDataModule
from glonetLit import Glonet

# Set up logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> float:
    """Main training function"""
    
    ##################################################################################################################-MODEL
    model = Glonet(cfg)

    ##################################################################################################################-DATA
    data_module = GlonetDataModule(cfg)

    ##################################################################################################################-TRAINING
    # Print configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.training.callbacks.monitor,  # Monitor epoch-level train loss
            dirpath=cfg.training.callbacks.dirpath,
            filename=cfg.training.callbacks.filename,
            save_top_k=cfg.training.callbacks.save_top_k,
            mode=cfg.training.callbacks.mode,
            save_last=True,
            # every_n_epochs=10,  # Save every epoch
            verbose=True  # Print when checkpoints are saved
        )
    ]
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir = cfg.training.tensorboard.save_dir,
        name=cfg.training.tensorboard.name,
        version=cfg.training.tensorboard.version
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.training.trainer.epochs,
        callbacks=callbacks,
        accelerator=cfg.training.trainer.accelerator,
        devices=cfg.training.trainer.devices,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        # gradient_clip_val=cfg.training.trainer.grad_clip_norm,
        precision=cfg.training.trainer.precision,
        num_sanity_val_steps=cfg.training.trainer.num_sanity_val_steps,
        fast_dev_run=cfg.training.trainer.fast_dev_run,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
