import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import logging 
# Set up logging
log = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from optimIC_GD_glonetLit import GlonetGradientInitialCondition
from optimIC_GD_dataset import GlorysDataModule



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig):
    # Configuration for multiple ocean states

    # =============================================================================================== Model
    # =====================================================================================================
    
    model = GlonetGradientInitialCondition(cfg)
    
    
    # =============================================================================================== Data
    # ====================================================================================================

    data_module = GlorysDataModule(cfg)


    # =============================================================================================== Training
    # ========================================================================================================
    
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
    if cfg.training.resume.enable_checkpoint :
        checkpoint_path = cfg.training.resume.checkpoint_path
        log.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, datamodule=data_module)
    
if __name__ == "__main__":
    main()