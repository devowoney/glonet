#!/usr/bin/env python3
"""
Simple example demonstrating multi-state ocean optimization.
Run this script to see how to optimize different ocean states.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_single_state():
    """Demo: Optimize a single ocean state"""
    print("=== Demo: Single State Optimization ===")
    
    # This is equivalent to running traning.py with state_number = 1
    from traning import main
    
    # You can edit traning.py to change the state_number
    print("This will optimize the state configured in traning.py")
    print("Edit traning.py to change 'state_number' to 1, 2, or 3")
    
    try:
        model, optimized_input, original_input, target = main()
        print("✓ Single state optimization completed!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def demo_all_states():
    """Demo: Optimize all three ocean states"""
    print("\n=== Demo: All States Optimization ===")
    
    from multi_state_optimization import optimize_all_states
    
    data_dir = "path/to/your/data"  # Update this path
    
    print("This will optimize all three ocean states:")
    print("- State 1: input1.nc with glonet_p1.pt")
    print("- State 2: input2.nc with glonet_p2.pt") 
    print("- State 3: input3.nc with glonet_p3.pt")
    
    try:
        results = optimize_all_states(data_dir, sample_index=0, max_epochs=10)  # Reduced epochs for demo
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"✓ Completed optimization for {successful}/3 states!")
        return results
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def demo_compare_states():
    """Demo: Compare optimization performance across states"""
    print("\n=== Demo: Compare States Performance ===")
    
    from ocean_states_optimizer import compare_states_optimization
    
    data_dir = "path/to/your/data"  # Update this path
    
    print("This will optimize all states and compare their performance...")
    
    try:
        comparison = compare_states_optimization(data_dir, sample_index=0, max_epochs=5)  # Very short for demo
        return comparison
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def demo_specific_states():
    """Demo: Optimize only specific states"""
    print("\n=== Demo: Specific States Only ===")
    
    from ocean_states_optimizer import run_specific_states
    
    data_dir = "path/to/your/data"  # Update this path
    
    # Let's say we only want to optimize states 1 and 3
    states_to_run = [1, 3]
    
    print(f"This will optimize only states: {states_to_run}")
    
    try:
        results = run_specific_states(data_dir, states_to_run=states_to_run, sample_index=0, max_epochs=10)
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"✓ Completed optimization for {successful}/{len(states_to_run)} requested states!")
        return results
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    """Run all demos"""
    print("=== Multi-State Ocean Optimization Demo ===")
    print("This demonstrates different ways to optimize ocean states.\n")
    
    print("Available demos:")
    print("1. Single state optimization (using traning.py)")
    print("2. All states optimization")
    print("3. Compare states performance")
    print("4. Specific states only")
    print("\nNOTE: Update 'data_dir' paths in the demo functions before running!")
    
    # You can uncomment any of these to run the demos:
    
    # Demo 1: Single state (requires proper configuration in traning.py)
    # demo_single_state()
    
    # Demo 2: All states
    # demo_all_states()
    
    # Demo 3: Compare states
    # demo_compare_states()
    
    # Demo 4: Specific states
    # demo_specific_states()
    
    print("\nTo run a demo, uncomment the corresponding line in this script.")
    print("Make sure to update the 'data_dir' path to point to your data directory.")

if __name__ == "__main__":
    main()
