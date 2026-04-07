#!/usr/bin/env python3
"""
Ensemble Analysis: Best Member Selection

This script calculates distance (RMSE) from ground truth for each ensemble member
and selects the best performing members based on lowest RMSE.

Methodology:
1. Load ensemble members from forecast directories
2. Load ground truth (real state at time=2)
3. Compute RMSE for all individual members
4. Select best N members based on lowest RMSE
5. Save best members and their forecasts
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path





def load_ensemble_members(directory, verbose=False):
    """
    Load all NetCDF files from a directory as ensemble members.
    
    Args:
        directory: Path to directory containing ensemble members
        verbose: Print detailed loading information
        
    Returns:
        List of xarray datasets
    """
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return []
    
    # Find all .nc files
    nc_files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    
    if not nc_files:
        print(f"Warning: No NetCDF files found in {directory}")
        return []
    
    print(f"Loading {len(nc_files)} files from {directory}...")
    members = []
    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file)
            members.append(ds)
            if verbose:
                print(f"  ✓ Loaded: {os.path.basename(nc_file)}")
        except Exception as e:
            print(f"  ✗ Error loading {nc_file}: {e}")
    
    print(f"Total members loaded: {len(members)}")
    return members


def load_real_state(real_state_path, time_index=2, verbose=False):
    """
    Load the ground truth state from combined_input.nc.
    
    Args:
        real_state_path: Directory containing combined_input.nc
        time_index: Time index to extract
        verbose: Print detailed information
        
    Returns:
        xarray Dataset with ground truth at specified time
    """
    print(f"\nLoading real state from: {real_state_path}")
    try:
        # Load the combined input file
        real_state_ds = xr.open_dataset(real_state_path, 
                                        # os.path.join(real_state_path, "combined_input.nc"), 
                                        chunks=96
        )
        
        # Extract specified time index
        real_state = real_state_ds.isel(time=time_index).persist()
        
        print(f"✓ Real state loaded successfully")
        if verbose:
            print(f"  Shape: {real_state['data'].shape}")
            print(f"  Variables: {list(real_state.data_vars)}")
            print(f"  Coordinates: {list(real_state.coords)}")
        
        return real_state
        
    except Exception as e:
        print(f"✗ Error loading real state: {e}")
        return None


def extract_time_zero(members, verbose=False):
    """
    Extract time=0 from each ensemble member.
    
    Args:
        members: List of xarray datasets
        verbose: Print detailed information
        
    Returns:
        List of datasets at time=0
    """
    print("\nExtracting time=0 from ensemble members...")
    
    t0_list = []
    for i, member in enumerate(members):
        try:
            # Extract time=0 (assuming 'time' dimension exists)
            if 'time' in member.dims:
                t0 = member.isel(time=0)
            else:
                t0 = member
            
            t0_list.append(t0)
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(members)} members...")
                
        except Exception as e:
            print(f"  ✗ Member {i+1}: error - {e}")
    
    print(f"Total time=0 slices extracted: {len(t0_list)}")
    return t0_list


def compute_ensemble_statistics(members_list, real_state, verbose=False):
    """
    Compute mean and variance across ensemble members.
    
    Args:
        members_list: List of xarray datasets (ensemble members at time=0)
        real_state: Ground truth dataset
        verbose: Print detailed statistics
        
    Returns:
        dict with 'mean', 'variance', 'std' datasets
    """
    if not members_list:
        print("No members to compute statistics")
        return None
    
    print(f"\nComputing ensemble statistics...")
    print(f"Number of members: {len(members_list)}")
    
    try:
        # Extract the data arrays and stack them
        data_arrays = []
        real_data = real_state['data'] if isinstance(real_state, xr.Dataset) and 'data' in real_state else real_state
        
        for i, member in enumerate(members_list):
            if 'data' in member:
                data_arrays.append((member['data'] - real_data).fillna(0))
            else:
                print(f"Warning: Member {i} doesn't have 'data' variable")
        
        if not data_arrays:
            print("No valid data arrays found")
            return None
        
        # Concatenate along new 'member' dimension
        stacked = xr.concat(data_arrays, dim='member')
        
        # Compute overall statistics
        ensemble_mean = stacked.mean(dim='member', skipna=True)
        ensemble_var = stacked.var(dim='member', skipna=True)
        ensemble_std = stacked.std(dim='member', skipna=True)
        
        print(f"✓ Statistics computed")
        
        if verbose:
            print(f"  Shape: {ensemble_mean.sizes}")
            print(f"  Mean value range: [{float(ensemble_mean.min()):.4f}, {float(ensemble_mean.max()):.4f}]")
            print(f"  Std dev value range: [{float(ensemble_std.min()):.6f}, {float(ensemble_std.max()):.6f}]")
            
            # Determine channel dimension name
            channel_dim = None
            for dim in ['ch', 'channel']:
                if dim in ensemble_mean.dims:
                    channel_dim = dim
                    break
            
            # Per-channel statistics
            if channel_dim is not None:
                n_channels = ensemble_mean.sizes[channel_dim]
                print(f"\n  Per-Channel Statistics ({n_channels} channels):")
                print(f"  {'Channel':<10} {'Mean':<15} {'Std Dev':<15}")
                print(f"  {'-'*40}")
                
                for ch in range(n_channels):
                    ch_mean = ensemble_mean.isel({channel_dim: ch})
                    ch_std = ensemble_std.isel({channel_dim: ch})
                    
                    mean_val = float(ch_mean.mean(skipna=True))
                    std_val = float(ch_std.mean(skipna=True))
                    
                    print(f"  {ch:<10} {mean_val:<15.6f} {std_val:<15.6f}")
        
        return {
            'mean': ensemble_mean,
            'variance': ensemble_var,
            'std': ensemble_std,
            'members': stacked
        }
    
    except Exception as e:
        print(f"✗ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_std_deviation(stats, ensemble_name, output_dir):
    """
    Visualize the ensemble standard deviation for 5 surface variables.
    
    Args:
        stats: Statistics dictionary from compute_ensemble_statistics
        ensemble_name: Name of the ensemble for plot title
        output_dir: Directory to save the figure (default: current directory)
    """
    # Define the 5 surface variables
    surface_vars = [
        ('SSH', 0, 'Sea Surface Height'),
        ('SST', 1, 'Sea Surface Temperature'),
        ('SSS', 2, 'Sea Surface Salinity'),
        ('uo', 3, 'Surface Zonal Velocity'),
        ('vo', 4, 'Surface Meridional Velocity')
    ]
    
    print(f"\nVisualizing standard deviation for {ensemble_name} ensemble")
    
    std = stats['std']
    
    # Determine channel dimension name
    channel_dim = None
    for dim in ['ch', 'channel']:
        if dim in std.dims:
            channel_dim = dim
            break
    
    if channel_dim is None:
        print(f"Warning: No channel dimension found for {ensemble_name}")
        return
    
    # Check if we have lat/lon coordinates
    has_spatial = 'lat' in std.dims and 'lon' in std.dims
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (var_name, ch_idx, var_desc) in enumerate(surface_vars):
        ax = axes[idx]
        
        # Extract standard deviation for this channel
        var_std = std.isel({channel_dim: ch_idx})
        
        # Compute statistics
        var_min = float(var_std.min(skipna=True))
        var_max = float(var_std.max(skipna=True))
        var_mean = float(var_std.mean(skipna=True))
        
        if has_spatial:
            # Spatial plot
            lats = var_std['lat'].values
            lons = var_std['lon'].values
            
            im = ax.pcolormesh(lons, lats, var_std.values,
                              cmap='viridis', shading='auto')
            
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label(f'Std Dev', fontsize=9)
            
            ax.set_title(f'{var_name}: {var_desc}\nMean: {var_mean:.3e}, Range: [{var_min:.3e}, {var_max:.3e}]', 
                        fontsize=11, fontweight='bold')
        else:
            # Histogram fallback
            var_data_flat = var_std.values.flatten()
            var_data_flat = var_data_flat[~np.isnan(var_data_flat)]
            
            ax.hist(var_data_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(var_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {var_mean:.3e}')
            ax.set_xlabel('Std Dev', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{var_name}: {var_desc}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    fig.suptitle(f'Ensemble Standard Deviation: {ensemble_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{ensemble_name}_std_deviation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def compute_member_mse(member, real_state):
    """
    Compute MSE between a single member and real state.
    
    Args:
        member: xarray Dataset (single ensemble member at time=0)
        real_state: xarray Dataset (ground truth)
    
    Returns:
        float: MSE value
    """
    # Get the 'data' variable
    if isinstance(member, xr.Dataset):
        member_data = member['data'] if 'data' in member else member
    else:
        member_data = member
        
    if isinstance(real_state, xr.Dataset):
        real_data = real_state['data'] if 'data' in real_state else real_state
    else:
        real_data = real_state
    
    # Compute MSE (skipna=True to handle land mask)
    diff = member_data - real_data
    mse = float((diff ** 2).mean(skipna=True))
    
    return mse


def compute_all_member_mse(members_t0, real_state, verbose=False):
    """
    Compute MSE for all individual ensemble members.
    
    Args:
        members_t0: List of ensemble members at time=0
        real_state: Ground truth dataset
        verbose: Print detailed progress
        
    Returns:
        List of dictionaries with member info and MSE/RMSE
    """
    print("\n" + "="*80)
    print("COMPUTING MSE FOR ALL INDIVIDUAL MEMBERS")
    print("="*80)
    
    all_member_results = []
    
    for idx, member in enumerate(members_t0):
        try:
            mse = compute_member_mse(member, real_state)
            rmse = np.sqrt(mse)
            
            # Store results
            result = {
                'member_idx': idx,
                'mse': mse,
                'rmse': rmse,
                'member_data': member
            }
            all_member_results.append(result)
            
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(members_t0)} members...")
            
        except Exception as e:
            print(f"  ✗ Error processing member {idx}: {e}")
    
    print(f"✓ Total members processed: {len(all_member_results)}")
    
    return all_member_results


def select_best_members(all_member_results, n_best=10, verbose=False):
    """
    Select the best N members based on lowest MSE.
    
    Args:
        all_member_results: List of member result dictionaries
        n_best: Number of best members to select
        verbose: Print detailed information
        
    Returns:
        List of best member result dictionaries
    """
    # Sort by MSE (ascending - best performers first)
    all_member_results.sort(key=lambda x: x['mse'])
    
    # Select top N best members
    best_members = all_member_results[:n_best]
    
    print(f"\n{'='*80}")
    print(f"TOP {n_best} BEST PERFORMING MEMBERS (Lowest MSE)")
    print("="*80)
    print(f"{'Rank':<6} {'Member':<10} {'MSE':<18} {'RMSE':<18}")
    print("-"*80)
    
    for rank, result in enumerate(best_members, 1):
        print(f"{rank:<6} {result['member_idx']:<10} "
              f"{result['mse']:<18.10e} {result['rmse']:<18.10e}")
    
    if verbose and len(all_member_results) > 0:
        all_mse_values = [r['mse'] for r in all_member_results]
        print(f"\n{'='*80}")
        print("MSE DISTRIBUTION STATISTICS")
        print("="*80)
        print(f"Total members: {len(all_mse_values)}")
        print(f"Mean MSE:      {np.mean(all_mse_values):.10e}")
        print(f"Median MSE:    {np.median(all_mse_values):.10e}")
        print(f"Std Dev MSE:   {np.std(all_mse_values):.10e}")
        print(f"Min MSE:       {np.min(all_mse_values):.10e} (best)")
        print(f"Max MSE:       {np.max(all_mse_values):.10e} (worst)")
    
    return best_members


def visualize_mse_distribution(all_member_results, best_members, output_dir):
    """
    Create visualization of MSE distribution and best members.
    
    Args:
        all_member_results: List of all member results
        best_members: List of best member results
        output_dir: Directory to save the figure (default: current directory)
    """
    all_mse_values = [r['mse'] for r in all_member_results]
    best_mse = [r['mse'] for r in best_members]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of all MSE values
    axes[0].hist(all_mse_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(best_mse[-1], color='red', linestyle='--', linewidth=2, label=f'Top {len(best_members)} threshold')
    axes[0].axvline(np.mean(all_mse_values), color='green', linestyle='--', linewidth=2, label='Mean MSE')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('MSE Distribution Across All Members', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sorted MSE values with best highlighted
    sorted_indices = np.arange(len(all_mse_values))
    axes[1].plot(sorted_indices, all_mse_values, 'o-', alpha=0.6, markersize=4, label='All members')
    axes[1].plot(sorted_indices[:len(best_members)], best_mse, 'ro', markersize=8, label=f'Top {len(best_members)} best')
    axes[1].set_xlabel('Rank (sorted by MSE)', fontsize=12)
    axes[1].set_ylabel('MSE', fontsize=12)
    axes[1].set_title('Ranked Members by MSE Performance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mse_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def save_best_members(best_members, forecast_dir, output_dir, forecast_cycle, all_member_results):
    """
    Save the best N members to output directory.
    
    Args:
        best_members: List of best member result dictionaries
        forecast_dir: Source directory containing original forecasts
        output_dir: Destination directory for best members
        forecast_cycle: Forecast cycle length (in days)
        all_member_results: List of all member results (for statistics)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"SAVING BEST {len(best_members)} MEMBERS")
    print("="*80)
    print(f"Output directory: {output_dir}\n")
    
    # Save the full forecast files for each of the best members
    for rank, result in enumerate(best_members, 1):
        member_idx = result['member_idx']
        
        # Get the original file path
        original_file = os.path.join(os.path.join(forecast_dir, os.pardir), 
                                      f"member_{member_idx:03d}_initial_condition.nc")
        forecast_file = os.path.join(forecast_dir, 
                                      f"member_{member_idx:03d}_initial_condition_forecast_{forecast_cycle}days.nc")
        
        # Create output filename
        output_file = os.path.join(output_dir, 
                                    f"rank{rank:02d}_member_{member_idx:03d}_mse_{result['mse']:.6e}.nc")

        try:
            # Load the full forecast (all timesteps, not just time=0)
            full_original = xr.open_dataset(original_file)
            full_forecast = xr.open_dataset(original_file)
            
            # Save to new location
            full_forecast.to_netcdf(output_file)
            full_forecast.to_netcdf(output_file)
            
            print(f"✓ Rank {rank}: member_{member_idx:03d} → {os.path.basename(output_file)}")
            
        except Exception as e:
            print(f"✗ Error saving member {member_idx}: {e}")
    
    # Save metadata summary
    metadata_file = os.path.join(output_dir, 'best_members_summary.txt')
    all_mse_values = [r['mse'] for r in all_member_results]
    
    with open(metadata_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TOP {len(best_members)} BEST PERFORMING ENSEMBLE MEMBERS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Selection criterion: Lowest MSE to ground truth\n")
        f.write(f"Ground truth: time index from combined_input.nc\n")
        f.write(f"Source directory: {forecast_dir}\n\n")
        f.write(f"{'Rank':<6} {'Member':<10} {'MSE':<20} {'RMSE':<20}\n")
        f.write("-"*80 + "\n")
        
        for rank, result in enumerate(best_members, 1):
            f.write(f"{rank:<6} {result['member_idx']:<10} {result['mse']:<20.10e} {result['rmse']:<20.10e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Best MSE:  {best_members[0]['mse']:.10e}\n")
        f.write(f"Worst (of top {len(best_members)}) MSE: {best_members[-1]['mse']:.10e}\n")
        f.write(f"Mean MSE (all {len(all_mse_values)}): {np.mean(all_mse_values):.10e}\n")
        f.write(f"Median MSE (all {len(all_mse_values)}): {np.median(all_mse_values):.10e}\n")
    
    print(f"\n✓ Metadata saved to: {metadata_file}")
    print("\n" + "="*80)
    print("SAVE COMPLETE")
    print("="*80)
    print(f"\nBest {len(best_members)} members saved to: {output_dir}")
    print(f"Total files: {len(best_members)} NetCDF files + 1 summary")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Select best performing ensemble members based on RMSE to ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--forecast-dir',
        type=str,
        required=True,
        help='Directory containing forecast ensemble members'
    )
    
    parser.add_argument(
        '--real-state-path',
        type=str,
        required=True,
        help='Path to ground truth data (combined_input.nc)'
    )
    
    parser.add_argument(
        '--time-index', '-t', 
        type=int,
        required=True,
        help='Time index for ground truth state'
    )
    
    parser.add_argument(
        '--n-best', '-n',
        type=int,
        required=True,
        help='Number of best members to select'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for best members (default: <forecast-dir>/best_N_members)'
    )

    parser.add_argument(
        '--forecast-cycle',
        type=int,
        default=7,
        help='Forecast cycle length (to match forecast output files name)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )
    
    parser.add_argument(
        '--compute-stats',
        action='store_true',
        help='Compute and display ensemble statistics'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.forecast_dir, f"best_{args.n_best}_members")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE MEMBER SELECTION")
    print("="*80)
    print(f"Forecast directory: {args.forecast_dir}")
    print(f"Ground truth path:  {args.real_state_path}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Number of best:     {args.n_best}")
    print(f"Time index:         {args.time_index}")
    print("="*80)
    
    # Check if forecast directory exists
    if not os.path.exists(args.forecast_dir):
        print(f"\nError: Forecast directory does not exist: {args.forecast_dir}")
        return 1
    
    # Load ensemble members
    members = load_ensemble_members(args.forecast_dir, verbose=args.verbose)
    if not members:
        print("\nError: No ensemble members loaded")
        return 1
    
    # Load ground truth
    real_state = load_real_state(args.real_state_path, time_index=args.time_index, verbose=args.verbose)
    if real_state is None:
        print("\nError: Failed to load ground truth")
        return 1
    
    # Extract time=0 from ensemble members
    members_t0 = extract_time_zero(members, verbose=args.verbose)
    if not members_t0:
        print("\nError: Failed to extract time=0 from members")
        return 1
    
    # Compute ensemble statistics (optional)
    if args.compute_stats:
        stats = compute_ensemble_statistics(members_t0, real_state, verbose=args.verbose)
        
        if stats is not None and args.visualize:
            visualize_std_deviation(stats, 'ensemble', output_dir=args.output_dir)
    
    # Compute MSE for all members
    all_member_results = compute_all_member_mse(members_t0, real_state, verbose=args.verbose)
    if not all_member_results:
        print("\nError: Failed to compute MSE for members")
        return 1
    
    # Select best N members
    best_members = select_best_members(all_member_results, n_best=args.n_best, verbose=args.verbose)
    if not best_members:
        print("\nError: Failed to select best members")
        return 1
    
    # Create visualizations (optional)
    if args.visualize:
        visualize_mse_distribution(all_member_results, best_members, output_dir=args.output_dir)
    
    # Save best members
    save_best_members(best_members, args.forecast_dir, args.output_dir, args.forecast_cycle, all_member_results)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETE")
    print("="*80)
    print(f"\nBest member IDs: {[r['member_idx'] for r in best_members]}")
    
    return 0


if __name__ == "__main__":
    exit(main())
