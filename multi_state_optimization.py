import torch
import torch.nn.functional as F
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from optimInitLit import Glonet, GlorysDataModule


def optimize_single_state(model_path, data_dir, input_file, sample_index=0, max_epochs=100):
    """
    Optimize a single ocean state.
    
    Args:
        model_path: Path to the trained model for this state
        data_dir: Directory containing the input files
        input_file: Which input file to use (e.g., "input1.nc")
        sample_index: Which sample to optimize
        max_epochs: Number of optimization epochs
    
    Returns:
        model, optimized_input, original_input, target
    """
    
    print(f"\n=== Optimizing {input_file} with {model_path} ===")
    
    # Initialize data module for this specific input file
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
    state_name = input_file.replace('.nc', '').replace('input', 'state')
    callbacks = [
        ModelCheckpoint(
            monitor="train_loss",
            dirpath=f"checkpoints/{state_name}/",
            filename="best-model-{epoch:02d}-{train_loss:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True
        )
    ]
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        precision=32,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False  # Disable to reduce output clutter
    )
    
    # Start training
    trainer.fit(model, data_module)
    
    # Save the optimized input sequence
    optimized_input = model.init_input.data.cpu()
    output_filename = f"optimized_{state_name}_sample_{sample_index}.pt"
    torch.save(optimized_input, output_filename)
    
    # Calculate and show the improvement
    original_input = input_sequence
    input_diff = torch.abs(optimized_input - original_input).mean()
    
    print(f"\n=== Results for {input_file} ===")
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


def optimize_all_states(data_dir, sample_index=0, max_epochs=100):
    """
    Optimize all three ocean states.
    
    Args:
        data_dir: Directory containing input1.nc, input2.nc, input3.nc
        sample_index: Which sample to optimize for each state
        max_epochs: Number of optimization epochs for each state
    
    Returns:
        Dictionary with results for each state
    """
    
    # Configuration for each ocean state
    states_config = {
        'state1': {
            'model_path': 'TrainedWeights/glonet_p1.pt',
            'input_file': 'input1.nc'
        },
        'state2': {
            'model_path': 'TrainedWeights/glonet_p2.pt',
            'input_file': 'input2.nc'
        },
        'state3': {
            'model_path': 'TrainedWeights/glonet_p3.pt',
            'input_file': 'input3.nc'
        }
    }
    
    results = {}
    
    print("=== Multi-State Ocean Optimization ===")
    print(f"Optimizing sample {sample_index} for all three ocean states")
    print(f"Data directory: {data_dir}")
    print(f"Max epochs per state: {max_epochs}\n")
    
    for state_name, config in states_config.items():
        try:
            model, optimized_input, original_input, target = optimize_single_state(
                model_path=config['model_path'],
                data_dir=data_dir,
                input_file=config['input_file'],
                sample_index=sample_index,
                max_epochs=max_epochs
            )
            
            results[state_name] = {
                'model': model,
                'optimized_input': optimized_input,
                'original_input': original_input,
                'target': target,
                'input_file': config['input_file'],
                'model_path': config['model_path']
            }
            
            print(f"✓ Successfully optimized {state_name}")
            
        except Exception as e:
            print(f"✗ Error optimizing {state_name}: {e}")
            results[state_name] = {'error': str(e)}
    
    # Summary
    print(f"\n=== Optimization Summary ===")
    successful = [k for k, v in results.items() if 'error' not in v]
    failed = [k for k, v in results.items() if 'error' in v]
    
    print(f"Successful optimizations: {len(successful)}")
    if successful:
        print(f"  - {', '.join(successful)}")
    
    if failed:
        print(f"Failed optimizations: {len(failed)}")
        print(f"  - {', '.join(failed)}")
    
    return results


def main():
    """
    Main function to run optimization for all ocean states.
    """
    # Configuration
    data_dir = "path/to/your/data"  # Update this path to your data directory
    sample_index = 0  # Which sample to optimize
    max_epochs = 100  # Number of optimization epochs per state
    
    # Run optimization for all states
    results = optimize_all_states(
        data_dir=data_dir,
        sample_index=sample_index,
        max_epochs=max_epochs
    )
    
    return results


if __name__ == "__main__":
    results = main()
