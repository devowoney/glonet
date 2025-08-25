import torch
import torch.nn.functional as F
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from optimInitLit import Glonet, GlorysDataModule

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig):
    # Configuration for multiple ocean states
    state_configs = {
        1: {'model_file': 'glonet_p1.pt', 'input_file': 'input1.nc'},
        2: {'model_file': 'glonet_p2.pt', 'input_file': 'input2.nc'},
        3: {'model_file': 'glonet_p3.pt', 'input_file': 'input3.nc'}
    }
    
    model_dir = cfg.model_dir
    data_dir = cfg.data_dir 
    state_number = cfg.state_number 
    sample_index = cfg.sample_index 
    
    if state_number not in state_configs:
        raise ValueError(f"Invalid state number: {state_number}. Choose 1, 2, or 3.")
    
    config = state_configs[state_number]
    model_path = model_dir + "/" + config['model_file']
    input_file = config['input_file']
    
    print(f"=== Optimizing Ocean State {state_number} ===")
    print(f"Model: {model_path}")
    print(f"Input file: {input_file}")
    print(f"Sample index: {sample_index}")
    
    # Initialize data module first to get the actual input sequence
    data_module = GlorysDataModule(
        data_dir=data_dir, 
        input_file=input_file,
        sample_index=sample_index, 
        batch_size=cfg.training.datamodule.batch_size, 
        num_workers=cfg.training.datamodule.num_workers
    )
    data_module.setup()

    # Get the input sequence and target for this sample
    input_sequence, target = data_module.train_dataset.input_sequence, data_module.train_dataset.target
    
    # Initialize model with the actual input sequence as the trainable parameter
    model = Glonet(model_path=model_path, cfg=cfg)
    
    # Move model and data to the correct device to avoid dtype/device mismatches
    device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(cfg.training.trainer, "devices", None) else "cpu")
    model.to(device)
    try:
        model.saved_model = model.saved_model.to(device)
    except Exception:
        pass
    input_sequence = input_sequence.to(device)
    target = target.to(device)
    
    # Reset gradients
    model.zero_grad()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.training.callbacks.monitor,
            dirpath=cfg.training.callbacks.dirpath,
            filename=cfg.training.callbacks.filename,
            save_top_k=cfg.training.callbacks.save_top_k,
            mode=cfg.training.callbacks.mode,
            save_last=True
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
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger
    )
    
    # Start training
    trainer.fit(model, data_module)
    
    # Save the optimized input sequence
    optimized_input = model.init_input.data.cpu()
    output_filename = f"optimized_state{state_number}_sample_{sample_index}.pt"
    torch.save(optimized_input.detach().cpu(), output_filename)
    
    # Calculate and show the improvement
    original_input = input_sequence
    input_diff = torch.abs(optimized_input - original_input).mean()
    
    print(f"\n=== Optimization Results for State {state_number} ===")
    print(f"Model: {model_path}")
    print(f"Input file: {input_file}")
    print(f"Sample index: {sample_index}")
    print(f"Optimized input saved to: {output_filename}")
    print(f"Input shape: {optimized_input.shape}")
    print(f"Original input stats: mean={original_input.mean():.4f}, std={original_input.std():.4f}")
    print(f"Optimized input stats: mean={optimized_input.mean():.4f}, std={optimized_input.std():.4f}")
    print(f"Mean absolute change: {input_diff:.6f}")
    print(f"Final loss: {trainer.logged_metrics.get('train_loss', 'N/A')}")
    
    # Test the optimized input vs original
    model.eval()
    with torch.no_grad():
        # Ensure tensors and model are on same device
        original_input_device = original_input.to(device)
        optimized_input_device = optimized_input.to(device)
        target_device = target.to(device)

        original_prediction = model.saved_model(original_input_device.unsqueeze(0))
        optimized_prediction = model.saved_model(optimized_input_device.unsqueeze(0))
        
        original_loss = F.mse_loss(original_prediction, target_device.unsqueeze(0))
        optimized_loss = F.mse_loss(optimized_prediction, target_device.unsqueeze(0))
        
        improvement = (original_loss - optimized_loss).item()
        improvement_percent = (improvement / original_loss.item()) * 100
        
        print(f"Original input loss: {original_loss.item():.6f}")
        print(f"Optimized input loss: {optimized_loss.item():.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
    
    return model, optimized_input, original_input, target


if __name__ == "__main__":
    main()