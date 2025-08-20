#!/usr/bin/env python3
"""
Utility script to easily run optimization for specific ocean states.
This provides simple functions to optimize any combination of the three states.
"""

import torch
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from multi_state_optimization import optimize_single_state, optimize_all_states

def optimize_state_1(data_dir, sample_index=0, max_epochs=100):
    """Optimize ocean state 1 (input1.nc with glonet_p1.pt)"""
    return optimize_single_state(
        model_path='TrainedWeights/glonet_p1.pt',
        data_dir=data_dir,
        input_file='input1.nc',
        sample_index=sample_index,
        max_epochs=max_epochs
    )

def optimize_state_2(data_dir, sample_index=0, max_epochs=100):
    """Optimize ocean state 2 (input2.nc with glonet_p2.pt)"""
    return optimize_single_state(
        model_path='TrainedWeights/glonet_p2.pt',
        data_dir=data_dir,
        input_file='input2.nc',
        sample_index=sample_index,
        max_epochs=max_epochs
    )

def optimize_state_3(data_dir, sample_index=0, max_epochs=100):
    """Optimize ocean state 3 (input3.nc with glonet_p3.pt)"""
    return optimize_single_state(
        model_path='TrainedWeights/glonet_p3.pt',
        data_dir=data_dir,
        input_file='input3.nc',
        sample_index=sample_index,
        max_epochs=max_epochs
    )

def compare_states_optimization(data_dir, sample_index=0, max_epochs=50):
    """
    Optimize all three states and compare their optimization performance.
    
    Args:
        data_dir: Directory containing input1.nc, input2.nc, input3.nc
        sample_index: Which sample to optimize
        max_epochs: Number of epochs for each state
    
    Returns:
        Dictionary with comparison results
    """
    
    print("=== Comparing Optimization Across All Ocean States ===")
    
    # Optimize all states
    results = optimize_all_states(data_dir, sample_index, max_epochs)
    
    # Compare results
    comparison = {}
    
    for state_name, result in results.items():
        if 'error' not in result:
            # Calculate improvement metrics
            model = result['model']
            optimized_input = result['optimized_input']
            original_input = result['original_input']
            target = result['target']
            
            model.eval()
            with torch.no_grad():
                original_pred = model.saved_model(original_input.unsqueeze(0))
                optimized_pred = model.saved_model(optimized_input.unsqueeze(0))
                
                original_loss = torch.nn.functional.mse_loss(original_pred, target.unsqueeze(0))
                optimized_loss = torch.nn.functional.mse_loss(optimized_pred, target.unsqueeze(0))
                
                improvement = (original_loss - optimized_loss).item()
                improvement_percent = (improvement / original_loss.item()) * 100
                
                input_change = torch.abs(optimized_input - original_input).mean().item()
                
                comparison[state_name] = {
                    'original_loss': original_loss.item(),
                    'optimized_loss': optimized_loss.item(),
                    'improvement': improvement,
                    'improvement_percent': improvement_percent,
                    'input_change': input_change,
                    'input_file': result['input_file']
                }
    
    # Print comparison
    print(f"\n=== Optimization Comparison (Sample {sample_index}) ===")
    print(f"{'State':<8} {'Input File':<12} {'Original Loss':<15} {'Optimized Loss':<16} {'Improvement':<12} {'% Improvement':<15} {'Input Change':<12}")
    print("-" * 100)
    
    for state_name, comp in comparison.items():
        print(f"{state_name:<8} {comp['input_file']:<12} {comp['original_loss']:<15.6f} {comp['optimized_loss']:<16.6f} {comp['improvement']:<12.6f} {comp['improvement_percent']:<15.2f} {comp['input_change']:<12.6f}")
    
    # Find best performing state
    if comparison:
        best_state = max(comparison.keys(), key=lambda k: comparison[k]['improvement_percent'])
        worst_state = min(comparison.keys(), key=lambda k: comparison[k]['improvement_percent'])
        
        print(f"\nBest optimization: {best_state} ({comparison[best_state]['improvement_percent']:.2f}% improvement)")
        print(f"Worst optimization: {worst_state} ({comparison[worst_state]['improvement_percent']:.2f}% improvement)")
    
    return comparison

def run_specific_states(data_dir, states_to_run=[1, 2, 3], sample_index=0, max_epochs=100):
    """
    Run optimization for specific states only.
    
    Args:
        data_dir: Directory containing input files
        states_to_run: List of states to optimize (e.g., [1, 3] to run only states 1 and 3)
        sample_index: Which sample to optimize
        max_epochs: Number of epochs per state
    
    Returns:
        Dictionary with results for requested states
    """
    
    state_functions = {
        1: optimize_state_1,
        2: optimize_state_2,
        3: optimize_state_3
    }
    
    results = {}
    
    print(f"=== Running Optimization for States: {states_to_run} ===")
    
    for state_num in states_to_run:
        if state_num in state_functions:
            print(f"\nOptimizing State {state_num}...")
            try:
                result = state_functions[state_num](data_dir, sample_index, max_epochs)
                results[f'state{state_num}'] = result
                print(f"✓ State {state_num} completed successfully")
            except Exception as e:
                print(f"✗ State {state_num} failed: {e}")
                results[f'state{state_num}'] = {'error': str(e)}
        else:
            print(f"Invalid state number: {state_num}")
    
    return results

# Example usage functions
def example_optimize_all():
    """Example: Optimize all three states"""
    data_dir = "path/to/your/data"  # Update this
    results = optimize_all_states(data_dir, sample_index=0, max_epochs=50)
    return results

def example_optimize_states_1_and_3():
    """Example: Optimize only states 1 and 3"""
    data_dir = "path/to/your/data"  # Update this
    results = run_specific_states(data_dir, states_to_run=[1, 3], sample_index=0, max_epochs=50)
    return results

def example_compare_all_states():
    """Example: Compare optimization performance across all states"""
    data_dir = "path/to/your/data"  # Update this
    comparison = compare_states_optimization(data_dir, sample_index=0, max_epochs=30)
    return comparison

if __name__ == "__main__":
    # You can run different examples by uncommenting them
    
    # Example 1: Optimize all states
    # results = example_optimize_all()
    
    # Example 2: Optimize only specific states
    # results = example_optimize_states_1_and_3()
    
    # Example 3: Compare all states
    # comparison = example_compare_all_states()
    
    # Example 4: Optimize just one state
    data_dir = "path/to/your/data"  # Update this
    model, optimized_input, original_input, target = optimize_state_1(data_dir, sample_index=0, max_epochs=100)
    
    print("Single state optimization completed!")
