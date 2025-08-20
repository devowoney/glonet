# Multi-State Ocean Optimization

This system allows you to optimize input sequences for all three ocean states in your Glonet system. Each state has its own input file and corresponding trained model.

## Ocean States Configuration

- **State 1**: `input1.nc` → `glonet_p1.pt`
- **State 2**: `input2.nc` → `glonet_p2.pt`  
- **State 3**: `input3.nc` → `glonet_p3.pt`

## Files Overview

- `traning.py`: Optimize a single state (configurable)
- `multi_state_optimization.py`: Core functions for multi-state optimization
- `ocean_states_optimizer.py`: Utility functions for easy optimization
- `src/optimInitLit.py`: Updated to handle multiple input files

## Usage Options

### 1. Optimize a Single State

Edit `traning.py` to choose which state:

```python
# In traning.py, change this line:
state_number = 1  # Change to 2 or 3 for other states
```

Then run:
```bash
python traning.py
```

### 2. Optimize All States at Once

```bash
python -c "from multi_state_optimization import optimize_all_states; optimize_all_states('path/to/your/data')"
```

### 3. Use Utility Functions

```python
from ocean_states_optimizer import optimize_state_1, optimize_state_2, optimize_state_3

# Optimize specific states
data_dir = "path/to/your/data"
model1, opt1, orig1, target1 = optimize_state_1(data_dir, sample_index=0)
model2, opt2, orig2, target2 = optimize_state_2(data_dir, sample_index=0)
model3, opt3, orig3, target3 = optimize_state_3(data_dir, sample_index=0)
```

### 4. Compare All States

```python
from ocean_states_optimizer import compare_states_optimization

comparison = compare_states_optimization("path/to/your/data", sample_index=0)
```

### 5. Optimize Specific States Only

```python
from ocean_states_optimizer import run_specific_states

# Optimize only states 1 and 3
results = run_specific_states("path/to/your/data", states_to_run=[1, 3])
```

## Configuration

Update these paths in the scripts:

- `data_dir`: Directory containing `input1.nc`, `input2.nc`, `input3.nc`
- Model paths are automatically set to:
  - `TrainedWeights/glonet_p1.pt`
  - `TrainedWeights/glonet_p2.pt`
  - `TrainedWeights/glonet_p3.pt`

## Output Files

For each optimized state, you get:

- **Optimized inputs**: `optimized_state{N}_sample_{sample_index}.pt`
- **Checkpoints**: Saved in `checkpoints/state{N}/`
- **Console output**: Shows optimization progress and results

## Example Output

```
=== Multi-State Ocean Optimization ===
Optimizing sample 0 for all three ocean states

=== Optimizing input1.nc with TrainedWeights/glonet_p1.pt ===
Loaded input1.nc successfully
Loaded sample 0 from input1.nc
Input sequence shape: torch.Size([2, 5, 64, 128])
...
Original input loss: 2.345678
Optimized input loss: 1.987654
Improvement: 0.358024 (15.28%)

=== Optimizing input2.nc with TrainedWeights/glonet_p2.pt ===
...

=== Optimization Comparison (Sample 0) ===
State    Input File   Original Loss   Optimized Loss    Improvement  % Improvement   Input Change
state1   input1.nc    2.345678        1.987654          0.358024     15.28           0.024567
state2   input2.nc    1.876543        1.654321          0.222222     11.84           0.019876
state3   input3.nc    3.456789        2.987654          0.469135     13.57           0.031234

Best optimization: state1 (15.28% improvement)
```

## Advanced Features

### Batch Processing Multiple Samples

```python
# Optimize multiple samples for each state
for sample_idx in range(5):  # First 5 samples
    results = optimize_all_states(data_dir, sample_index=sample_idx, max_epochs=50)
```

### Custom Configuration

```python
from multi_state_optimization import optimize_single_state

# Custom optimization with specific parameters
model, opt_input, orig_input, target = optimize_single_state(
    model_path='TrainedWeights/glonet_p1.pt',
    data_dir='your/data/dir',
    input_file='input1.nc',
    sample_index=5,
    max_epochs=200
)
```

### Performance Monitoring

The system automatically tracks:
- Training loss progression
- Input sequence changes
- Prediction improvements
- Optimization time per state

## Memory and Performance

- Each state is optimized sequentially to manage memory usage
- GPU memory is freed between state optimizations
- Checkpoints are saved separately for each state
- Progress bars show optimization status

## Troubleshooting

### Common Issues

1. **File not found**: Ensure your data directory contains `input1.nc`, `input2.nc`, `input3.nc`
2. **Model not found**: Verify the model files exist in `TrainedWeights/`
3. **Memory issues**: Reduce `max_epochs` or optimize states one at a time
4. **Shape mismatches**: All input files should have compatible shapes

### Debugging

1. **Test single state first**: Use `traning.py` with `state_number = 1`
2. **Check data loading**: Verify files load correctly
3. **Verify models**: Ensure model files are valid PyTorch models
4. **Check sample indices**: Make sure `sample_index` is within dataset range

## Expected Workflow

1. **Setup**: Ensure all input files and model files are available
2. **Test**: Run single state optimization first
3. **Scale**: Run multi-state optimization  
4. **Analyze**: Compare results across states
5. **Apply**: Use optimized inputs for better forecasting

This multi-state system allows you to systematically optimize and compare different ocean states, giving you insights into which states benefit most from input optimization.
