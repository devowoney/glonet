"""
Utility functions for mapping parameters between JIT and PyTorch models
when parameter names don't match exactly.
"""

import torch
import re
from collections import defaultdict

def analyze_parameter_structures(jit_model, pytorch_model):
    """Analyze and compare parameter structures between models."""
    
    print("=== JIT Model Structure Analysis ===")
    jit_params = list(jit_model.named_parameters())
    jit_shapes = defaultdict(list)
    
    for name, param in jit_params:
        shape_key = tuple(param.shape)
        jit_shapes[shape_key].append(name)
    
    print(f"JIT model has {len(jit_params)} parameters")
    print("Shape distribution:")
    for shape, names in sorted(jit_shapes.items()):
        print(f"  {shape}: {len(names)} parameters")
        if len(names) <= 3:  # Show examples for small groups
            for name in names:
                print(f"    - {name}")
    
    print("\n=== PyTorch Model Structure Analysis ===")
    pytorch_params = list(pytorch_model.named_parameters())
    pytorch_shapes = defaultdict(list)
    
    for name, param in pytorch_params:
        shape_key = tuple(param.shape)
        pytorch_shapes[shape_key].append(name)
    
    print(f"PyTorch model has {len(pytorch_params)} parameters")
    print("Shape distribution:")
    for shape, names in sorted(pytorch_shapes.items()):
        print(f"  {shape}: {len(names)} parameters")
        if len(names) <= 3:  # Show examples for small groups
            for name in names:
                print(f"    - {name}")
    
    return jit_shapes, pytorch_shapes

def copy_parameters_by_shape_order(jit_model, pytorch_model):
    """
    Copy parameters by matching shapes in the order they appear.
    This is a heuristic approach when parameter names don't match.
    """
    
    # Get parameters sorted by their order of appearance
    jit_params = [(name, param) for name, param in jit_model.named_parameters()]
    pytorch_params = [(name, param) for name, param in pytorch_model.named_parameters()]
    
    # Group by shape
    jit_by_shape = defaultdict(list)
    pytorch_by_shape = defaultdict(list)
    
    for name, param in jit_params:
        jit_by_shape[tuple(param.shape)].append((name, param))
    
    for name, param in pytorch_params:
        pytorch_by_shape[tuple(param.shape)].append((name, param))
    
    print("=== Attempting Parameter Copy by Shape Matching ===")
    copied_count = 0
    
    with torch.no_grad():
        for shape in pytorch_by_shape.keys():
            if shape in jit_by_shape:
                jit_group = jit_by_shape[shape]
                pytorch_group = pytorch_by_shape[shape]
                
                # Copy parameters in order within each shape group
                min_count = min(len(jit_group), len(pytorch_group))
                
                for i in range(min_count):
                    jit_name, jit_param = jit_group[i]
                    pytorch_name, pytorch_param = pytorch_group[i]
                    
                    pytorch_param.copy_(jit_param.to(pytorch_param.device).to(pytorch_param.dtype))
                    print(f"[SUCCESS] Copied {jit_name} -> {pytorch_name} (shape: {shape})")
                    copied_count += 1
                
                if len(jit_group) != len(pytorch_group):
                    print(f"[WARNING] Shape {shape}: JIT has {len(jit_group)} params, PyTorch has {len(pytorch_group)} params")
            else:
                print(f"[WARNING] No JIT parameters found for PyTorch shape {shape}")
                for pytorch_name, _ in pytorch_by_shape[shape]:
                    print(f"  - Missing: {pytorch_name}")
    
    print(f"\n[SUMMARY] Successfully copied {copied_count} parameters")
    return copied_count

def copy_jit_to_pytorch_enhanced(jit_model, pytorch_model):
    """
    Enhanced parameter copying that tries multiple strategies to match parameters.
    """
    
    print("=== Enhanced Parameter Copying ===")
    
    # First try: exact name matching
    jit_state_dict = {name: param for name, param in jit_model.named_parameters()}
    pytorch_state_dict = {name: param for name, param in pytorch_model.named_parameters()}
    
    copied_exact = 0
    for name, param in pytorch_state_dict.items():
        if name in jit_state_dict:
            jit_param = jit_state_dict[name]
            if param.shape == jit_param.shape:
                param.data.copy_(jit_param.data.to(param.device).to(param.dtype))
                print(f"[EXACT] Copied {name}")
                copied_exact += 1
    
    print(f"Exact name matches: {copied_exact}")
    
    # Second try: shape-based matching for remaining parameters
    if copied_exact < len(pytorch_state_dict):
        print("Attempting shape-based matching for remaining parameters...")
        return copy_parameters_by_shape_order(jit_model, pytorch_model)
    
    return copied_exact

def verify_model_equivalence(model1, model2, test_input):
    """Verify that two models produce similar outputs."""
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        try:
            output1 = model1(test_input)
            output2 = model2(test_input)
            
            if torch.allclose(output1, output2, atol=1e-5):
                print("[SUCCESS] ✓ Models produce identical outputs!")
                return True
            else:
                max_diff = torch.abs(output1 - output2).max().item()
                mean_diff = torch.abs(output1 - output2).mean().item()
                print(f"[INFO] Output differences - Max: {max_diff:.2e}, Mean: {mean_diff:.2e}")
                
                if max_diff < 1e-3:
                    print("[ACCEPTABLE] ✓ Models produce very similar outputs!")
                    return True
                else:
                    print("[WARNING] ✗ Models produce significantly different outputs!")
                    return False
        except Exception as e:
            print(f"[ERROR] Failed to test model equivalence: {e}")
            return False