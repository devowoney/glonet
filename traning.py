import torch
import torch.nn.functional as F
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from optimInitLit import Glonet, GlorysDataModule


def main():
    # Configuration for multiple ocean states
    state_configs = {
        1: {'model_path': 'TrainedWeights/glonet_p1.pt', 'input_file': 'input1.nc'},
        2: {'model_path': 'TrainedWeights/glonet_p2.pt', 'input_file': 'input2.nc'},
        3: {'model_path': 'TrainedWeights/glonet_p3.pt', 'input_file': 'input3.nc'}
    }
    
    # Choose which state to optimize (1, 2, or 3)
    state_number = 1  # Change this to 2 or 3 for other states
    data_dir = "path/to/your/data"  # Update this path to your data directory
    sample_index = 0  # Which sample from the dataset to optimize
    
    if state_number not in state_configs:
        raise ValueError(f"Invalid state number: {state_number}. Choose 1, 2, or 3.")
    
    config = state_configs[state_number]
    model_path = config['model_path']
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
        batch_size=1, 
        num_workers=0
    )
    data_module.setup()
    
    # Get the input sequence and target for this sample
    input_sequence, target = data_module.train_dataset.input_sequence, data_module.train_dataset.target
    
    print(f"Loaded sample {sample_index} from {input_file}")
    print(f"Input sequence shape: {input_sequence.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Input sequence stats: mean={input_sequence.mean():.4f}, std={input_sequence.std():.4f}")
    
    # Initialize model with the actual input sequence as the trainable parameter
    model = Glonet(model_path=model_path, init_input=input_sequence.clone())
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="train_loss",
            dirpath=f"checkpoints/state{state_number}/",
            filename="best-model-{epoch:02d}-{train_loss:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True
        )
    ]
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator="auto",  # Will use GPU if available
        devices="auto",
        log_every_n_steps=1,  # Log every step since we only have 1 sample
        gradient_clip_val=1.0,  # Gradient clipping for stability
        precision=32,  # Use 32-bit precision for initial condition optimization
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Start training
    trainer.fit(model, data_module)
    
    # Save the optimized input sequence
    optimized_input = model.init_input.data.cpu()
    output_filename = f"optimized_state{state_number}_sample_{sample_index}.pt"
    torch.save(optimized_input, output_filename)
    
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
        original_prediction = model.saved_model(original_input.unsqueeze(0))
        optimized_prediction = model.saved_model(optimized_input.unsqueeze(0))
        
        original_loss = F.mse_loss(original_prediction, target.unsqueeze(0))
        optimized_loss = F.mse_loss(optimized_prediction, target.unsqueeze(0))
        
        improvement = (original_loss - optimized_loss).item()
        improvement_percent = (improvement / original_loss.item()) * 100
        
        print(f"Original input loss: {original_loss.item():.6f}")
        print(f"Optimized input loss: {optimized_loss.item():.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")
    
    return model, optimized_input, original_input, target


if __name__ == "__main__":
    main()