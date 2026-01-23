import gc
import sys
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
import time
from torch.amp import autocast
import torch
import argparse
from pathlib import Path
from typing import Union, List, Optional
import glob
import logging

# Add current directory to path first to import local utility.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(1, '/Odyssey/private/j25lee/glonet/glonet_daily_forecast_local') # For utility.py

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
INPUT_LOCATION = "/Odyssey/public/glonet"
user = os.environ.get("USER")
DEFAULT_OUTPUT_LOCATION = f"/Odyssey/private/{user}/glonet/output"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


#####
## GPU Setup
#####

def setup_slurm_task():
    """Setup SLURM task-based parallelism (no DDP needed)"""
    if 'SLURM_PROCID' in os.environ:
        # Running with Slurm - simple task distribution
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Determine which GPU to use
        if 'SLURM_LOCALID' in os.environ:
            local_rank = int(os.environ['SLURM_LOCALID'])
        else:
            # Fallback: assume one task per GPU
            local_rank = rank
        
        # Ensure local_rank is within valid GPU range
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available")
        
        local_rank = local_rank % num_gpus
        
        # Set GPU device for this task
        torch.cuda.set_device(local_rank)
        
        logger.info(f"[Task {rank}/{world_size}] Using GPU {local_rank}")
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return 0, 1, 0
        else:
            raise RuntimeError("No GPU available")


#####
## Forecast Functions
#####

def make_nc(vars, denormalizer, ti, lead):
    vars = denormalizer(vars)
    d = xr.open_dataset(f"{MODEL_LOCATION}/ref1.nc")
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["zos"] = d["zos"] * vars.numpy()[0, :, 0:1].squeeze()
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 1:2]
    d["so"] = d["so"] * vars.numpy()[0, :, 2:3]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 3:4]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 4:5]
    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    )
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


def make_nc2(vars, denormalizer, ti, lead):
    vars = denormalizer(vars)
    d = xr.open_dataset(f"{MODEL_LOCATION}/ref2.nc")
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 0:10]
    d["so"] = d["so"] * vars.numpy()[0, :, 10:20]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 20:30]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 30:40]
    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    )
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


def make_nc3(vars, denormalizer, ti, lead):
    vars = denormalizer(vars)
    d = xr.open_dataset(f"{MODEL_LOCATION}/ref3.nc")
    d = xr.concat([d] * vars.shape[1], dim="time")
    d["thetao"] = d["thetao"] * vars.numpy()[0, :, 0:10]
    d["so"] = d["so"] * vars.numpy()[0, :, 10:20]
    d["uo"] = d["uo"] * vars.numpy()[0, :, 20:30]
    d["vo"] = d["vo"] * vars.numpy()[0, :, 30:40]
    time = np.arange(
        str(ti + timedelta(days=2 * lead)),
        str(ti + timedelta(days=2 * lead + 2)),
        dtype="datetime64[D]",
    )
    d = d.assign_coords(time=time)
    return xr.decode_cf(d)


def add_metadata(ds, date):
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    # Add global attributes
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["area"] = "Global"
    ds.attrs["contact"] = "glonet@mercator-ocean.eu"
    ds.attrs["institution"] = "Mercator Ocean International"
    ds.attrs["source"] = "MOI GLONET"
    ds.attrs["title"] = (
        "daily mean fields from GLONET 1/4 degree resolution Forecast updated Daily"
    )
    ds.attrs["references"] = "www.edito.eu"

    if "regrid_method" in ds.attrs:
        del ds.attrs["regrid_method"]

    # zos variable
    ds["zos"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "unit_long": "Meters",
        "units": "m",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # latitude variable
    ds["latitude"].attrs = {
        "axis": "Y",
        "long_name": "Latitude",
        "standard_name": "latitude",
        "step": ds.latitude.values[1] - ds.latitude.values[0],
        "unit_long": "Degrees North",
        "units": "degrees_north",
        "valid_max": ds.latitude.values.max(),
        "valid_min": ds.latitude.values.min(),
    }

    # longitude variable
    ds["longitude"].attrs = {
        "axis": "X",
        "long_name": "Longitude",
        "standard_name": "longitude",
        "step": ds.longitude.values[1] - ds.longitude.values[0],
        "unit_long": "Degrees East",
        "units": "degrees_east",
        "valid_max": ds.longitude.values.max(),
        "valid_min": ds.longitude.values.min(),
    }

    # time variable
    ds["time"].attrs = {
        "valid_min": str(date + timedelta(days=1)),
        "valid_max": str(date + timedelta(days=10)),
    }

    # depth variable
    ds["depth"].attrs = {
        "axis": "Z",
        "long_name": "Elevation",
        "positive": "down",
        "standard_name": "elevation",
        "unit_long": "Meters",
        "units": "m",
        "valid_min": 0.494025,
        "valid_max": 5727.917,
    }

    # uo variable
    ds["uo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Eastward velocity",
        "standard_name": "eastward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # vo variable
    ds["vo"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Northward velocity",
        "standard_name": "northward_sea_water_velocity",
        "unit_long": "Meters per second",
        "units": "m s-1",
        "valid_max": 5.0,
        "valid_min": -5.0,
    }

    # so variable
    ds["so"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Salinity",
        "standard_name": "sea_water_salinity",
        "unit_long": "Practical Salinity Unit",
        "units": "1e-3",
        "valid_max": 50.0,
        "valid_min": 0.0,
    }

    # thetao variable
    ds["thetao"].attrs = {
        "cell_methods": "area: mean",
        "long_name": "Temperature",
        "standard_name": "sea_water_potential_temperature",
        "unit_long": "Degrees Celsius",
        "units": "degrees_C",
        "valid_max": 40.0,
        "valid_min": -10.0,
    }
    return ds


def aforecast(d, date, cycle: int, device):
    """Process single member forecast for part 1"""
    from utility import get_denormalizer1, get_normalizer1

    denormalizer = get_denormalizer1(MODEL_LOCATION)
    normalizer = get_normalizer1(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.to(device).unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.to(device).unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()
    
    for i in range(1, int((cycle + 1) / 2) + 1):
        model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p1.pt")
        model_inf = model_inf.to(device)
        
        with torch.no_grad():
            model_inf.eval()
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar.to(device)
        del outvar, model_inf
        gc.collect()
        torch.cuda.empty_cache()
        
    del vin, mask
    gc.collect()
    return datasets


def aforecast_batch(d_batch, dates, cycle: int, device):
    """Process batch of members forecast for part 1"""
    from utility import get_denormalizer1, get_normalizer1

    denormalizer = get_denormalizer1(MODEL_LOCATION)
    normalizer = get_normalizer1(MODEL_LOCATION)
    
    batch_size = d_batch.sizes['batch']
    
    # Create masks for the batch
    nan_mask = np.isnan(d_batch.data[:, 1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32).to(device)
    
    # Prepare input batch
    data = np.nan_to_num(d_batch.data.values, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    # Apply normalization to each member in the batch
    vin = torch.stack([normalizer(vin[b]) for b in range(batch_size)])
    vin = vin.to(device).contiguous()
    
    all_datasets = [[] for _ in range(batch_size)]
    del data, nan_mask
    gc.collect()
    
    # Load model once for the batch
    model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p1.pt")
    model_inf = model_inf.to(device)
    model_inf.eval()
    
    for i in range(1, int((cycle + 1) / 2) + 1):
        with torch.no_grad():
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar_cpu = outvar.detach().cpu()
        
        # Process each member in the batch
        for b in range(batch_size):
            d = make_nc(outvar_cpu[b:b+1], denormalizer, dates[b] - timedelta(days=1), i)
            all_datasets[b].append(d)
        
        vin = outvar
        del outvar_cpu
        gc.collect()
    
    del vin, mask, model_inf
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_datasets


def aforecast2(d, date, cycle, device):
    """Process single member forecast for part 2"""
    from utility import get_denormalizer2, get_normalizer2

    denormalizer = get_denormalizer2(MODEL_LOCATION)
    normalizer = get_normalizer2(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.to(device).unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.to(device).unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()

    for i in range(1, int((cycle + 1) / 2) + 1):
        model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p2.pt")
        model_inf = model_inf.to(device)
        
        with torch.no_grad():
            model_inf.eval()
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc2(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar.to(device)
        del outvar, model_inf
        gc.collect()
        torch.cuda.empty_cache()
        
    del vin, mask
    gc.collect()
    return datasets


def aforecast2_batch(d_batch, dates, cycle: int, device):
    """Process batch of members forecast for part 2"""
    from utility import get_denormalizer2, get_normalizer2

    denormalizer = get_denormalizer2(MODEL_LOCATION)
    normalizer = get_normalizer2(MODEL_LOCATION)
    
    batch_size = d_batch.sizes['batch']
    
    # Create masks for the batch
    nan_mask = np.isnan(d_batch.data[:, 1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32).to(device)
    
    # Prepare input batch
    data = np.nan_to_num(d_batch.data.values, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    # Apply normalization to each member in the batch
    vin = torch.stack([normalizer(vin[b]) for b in range(batch_size)])
    vin = vin.to(device).contiguous()
    
    all_datasets = [[] for _ in range(batch_size)]
    del data, nan_mask
    gc.collect()
    
    # Load model once for the batch
    model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p2.pt")
    model_inf = model_inf.to(device)
    model_inf.eval()
    
    for i in range(1, int((cycle + 1) / 2) + 1):
        with torch.no_grad():
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar_cpu = outvar.detach().cpu()
        
        # Process each member in the batch
        for b in range(batch_size):
            d = make_nc2(outvar_cpu[b:b+1], denormalizer, dates[b] - timedelta(days=1), i)
            all_datasets[b].append(d)
        
        vin = outvar
        del outvar_cpu
        gc.collect()
    
    del vin, mask, model_inf
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_datasets


def aforecast3(d, date, cycle, device):
    """Process single member forecast for part 3"""
    from utility import get_denormalizer3, get_normalizer3

    denormalizer = get_denormalizer3(MODEL_LOCATION)
    normalizer = get_normalizer3(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.to(device).unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.to(device).unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()

    for i in range(1, int((cycle + 1) / 2) + 1):
        model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p3.pt")
        model_inf = model_inf.to(device)
        
        with torch.no_grad():
            model_inf.eval()
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc3(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar.to(device)
        del outvar, model_inf
        gc.collect()
        torch.cuda.empty_cache()
        
    del vin, mask
    gc.collect()
    return datasets


def aforecast3_batch(d_batch, dates, cycle: int, device):
    """Process batch of members forecast for part 3"""
    from utility import get_denormalizer3, get_normalizer3

    denormalizer = get_denormalizer3(MODEL_LOCATION)
    normalizer = get_normalizer3(MODEL_LOCATION)
    
    batch_size = d_batch.sizes['batch']
    
    # Create masks for the batch
    nan_mask = np.isnan(d_batch.data[:, 1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32).to(device)
    
    # Prepare input batch
    data = np.nan_to_num(d_batch.data.values, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    # Apply normalization to each member in the batch
    vin = torch.stack([normalizer(vin[b]) for b in range(batch_size)])
    vin = vin.to(device).contiguous()
    
    all_datasets = [[] for _ in range(batch_size)]
    del data, nan_mask
    gc.collect()
    
    # Load model once for the batch
    model_inf = torch.jit.load(MODEL_LOCATION + "/" + "glonet_p3.pt")
    model_inf = model_inf.to(device)
    model_inf.eval()
    
    for i in range(1, int((cycle + 1) / 2) + 1):
        with torch.no_grad():
            with autocast(device_type="cuda"):
                vin = vin * mask
                outvar = model_inf(vin)
                outvar_cpu = outvar.detach().cpu()
        
        # Process each member in the batch
        for b in range(batch_size):
            d = make_nc3(outvar_cpu[b:b+1], denormalizer, dates[b] - timedelta(days=1), i)
            all_datasets[b].append(d)
        
        vin = outvar
        del outvar_cpu
        gc.collect()
    
    del vin, mask, model_inf
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_datasets


def load_batch_data(member_files: List[Path]) -> tuple:
    """Load multiple ensemble members into batched xarray datasets"""
    batch_data = []
    dates = []
    member_names = []
    
    for member_file in member_files:
        rdata = xr.open_dataset(member_file)
        batch_data.append(rdata)
        date = rdata.time.data[1].astype("M8[D]").astype(datetime)
        dates.append(date)
        member_names.append(member_file.stem)
    
    # Concatenate along a new batch dimension
    batch_dataset = xr.concat(batch_data, dim='batch')
    
    return batch_dataset, dates, member_names


def create_forecast_batch(member_files: List[Path],
                         forecast_cycle: int,
                         output_path: str,
                         device,
                         rank: int) -> List[xr.Dataset]:
    """Create forecasts for a batch of ensemble members"""
    
    batch_size = len(member_files)
    
    logger.info(f"[Task {rank}, GPU {device}] Processing batch of {batch_size} members...")
    
    # Load batch data
    batch_dataset, dates, member_names = load_batch_data(member_files)
    
    start_datetime = str(dates[0] - timedelta(days=1))
    end_datetime = str(dates[0] + timedelta(days=forecast_cycle))
    
    logger.info(f"[Task {rank}, GPU {device}] Creating forecasts from {start_datetime} to {end_datetime}...")

    start_time = time.time()
    
    # Split data into three parts for each member
    rdata1_batch = batch_dataset.isel(ch=slice(0, 5))
    rdata2_batch = batch_dataset.isel(ch=slice(5, 45))
    rdata3_batch = batch_dataset.isel(ch=slice(45, 85))
    
    # Run batch forecasts
    ds1_batch = aforecast_batch(rdata1_batch, dates, cycle=forecast_cycle, device=device)
    del rdata1_batch
    gc.collect()
    
    ds2_batch = aforecast2_batch(rdata2_batch, dates, cycle=forecast_cycle, device=device)
    del rdata2_batch
    gc.collect()
    
    ds3_batch = aforecast3_batch(rdata3_batch, dates, cycle=forecast_cycle, device=device)
    del rdata3_batch
    gc.collect()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"[Task {rank}, GPU {device}] Batch forecast time: {execution_time:.4f} seconds ({execution_time/batch_size:.2f}s per member)")
    
    # Process each member in the batch
    results = []
    os.makedirs(output_path, exist_ok=True)
    
    for b in range(batch_size):
        # Combine results for this member
        combined1 = xr.concat(ds1_batch[b], dim="time")
        combined2 = xr.concat(ds2_batch[b], dim="time")
        combined3 = xr.concat(ds3_batch[b], dim="time")
        
        combined4 = xr.concat([combined1, combined2, combined3], dim="depth")
        combined4["zos"] = combined4.zos.isel(depth=0)
        combined4 = add_metadata(combined4, dates[b])
        
        vars_order = ['zos', 'thetao', 'so', 'uo', 'vo']
        combined4 = combined4[vars_order]
        
        # Save output
        output_file = os.path.join(output_path, f"{member_names[b]}_forecast_{forecast_cycle}days.nc")
        combined4.to_netcdf(output_file)
        
        logger.info(f"[Task {rank}, GPU {device}] Saved: {member_names[b]}")
        
        results.append(combined4)
        
        del combined1, combined2, combined3, combined4
        gc.collect()
    
    del ds1_batch, ds2_batch, ds3_batch
    gc.collect()
    
    return results


def create_forecast(rdata_path: Path,
                    forecast_cycle: int,
                    output_path: str,
                    device,
                    rank: int) -> xr.Dataset:
    """Create forecast for a single ensemble member"""
    
    # Extract member info
    member_name = rdata_path.stem  # e.g., member_000_initial_condition
    
    logger.info(f"[Task {rank}, GPU {device}] Processing {member_name}...")
    
    # Load data
    rdata = xr.open_dataset(rdata_path)
    date = rdata.time.data[1].astype("M8[D]").astype(datetime)
    
    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date + timedelta(days=forecast_cycle))
    
    logger.info(f"[Task {rank}, GPU {device}] Creating forecast from {start_datetime} to {end_datetime}...")

    start_time = time.time()
    
    # Split data
    rdata1 = rdata.isel(ch=slice(0, 5))
    rdata2 = rdata.isel(ch=slice(5, 45))
    rdata3 = rdata.isel(ch=slice(45, 85))
    
    # Run forecasts
    ds1 = aforecast(rdata1, date - timedelta(days=1), cycle=forecast_cycle, device=device)
    del rdata1
    gc.collect()
    
    ds2 = aforecast2(rdata2, date - timedelta(days=1), cycle=forecast_cycle, device=device)
    del rdata2
    gc.collect()
    
    ds3 = aforecast3(rdata3, date - timedelta(days=1), cycle=forecast_cycle, device=device)
    del rdata3
    gc.collect()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"[Task {rank}, GPU {device}] Forecast time: {execution_time:.4f} seconds")
    
    # Combine results
    combined1 = xr.concat(ds1, dim="time")
    combined2 = xr.concat(ds2, dim="time")
    combined3 = xr.concat(ds3, dim="time")
    del ds1, ds2, ds3
    gc.collect()

    combined4 = xr.concat([combined1, combined2, combined3], dim="depth")
    combined4["zos"] = combined4.zos.isel(depth=0)
    combined4 = add_metadata(combined4, date)
    
    vars_order = ['zos', 'thetao', 'so', 'uo', 'vo']
    combined4 = combined4[vars_order]
    
    del combined1, combined2, combined3
    gc.collect()

    # Save output
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{member_name}_forecast_{forecast_cycle}days.nc")
    combined4.to_netcdf(output_file)
    
    logger.info(f"[Task {rank}, GPU {device}] Forecast completed: {output_file}")
    
    return combined4


def get_ensemble_files(ensemble_dir: str, ensemble_type: str = "one_vector") -> List[Path]:
    """Get all ensemble member files from directory"""
    ensemble_path = Path(ensemble_dir) / ensemble_type
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory not found: {ensemble_path}")
    
    member_files = sorted(ensemble_path.glob("member_*_initial_condition.nc"))
    
    if not member_files:
        raise ValueError(f"No ensemble members found in {ensemble_path}")
    
    return member_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLONET Ensemble Forward - GPU Parallel Computing with Slurm"
    )
    
    parser.add_argument(
        "--ensemble-dir",
        type=str,
        default="/Odyssey/private/j25lee/ncprocessing/ensemble_glorys/ensemble_output_gpu",
        help="Path to ensemble directory"
    )
    
    parser.add_argument(
        "--ensemble-type",
        type=str,
        default="one_vector",
        choices=["one_vector", "pointwise_1", "pointwise_2"],
        help="Ensemble type subdirectory"
    )
    
    parser.add_argument(
        "-c", "--cycle",
        dest="forecast_cycle",
        type=int,
        default=7,
        help="Forecast cycle in days (default: 7)"
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str,
        default=None,
        help="Output directory path"
    )
    
    parser.add_argument(
        "--members-per-gpu",
        type=int,
        default=None,
        help="Number of members per GPU (default: auto-distribute)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for parallel processing on each GPU (default: 2)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup SLURM task distribution
    try:
        rank, world_size, local_rank = setup_slurm_task()
        device = torch.device(f"cuda:{local_rank}")
    except Exception as e:
        logger.error(f"Error: Could not setup GPU: {e}")
        logger.info("Falling back to single GPU mode...")
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"GLONET Ensemble Forecast - Task-Based Parallel Computing")
        logger.info(f"{'='*60}")
        logger.info(f"Total tasks/GPUs: {world_size}")
        logger.info(f"Ensemble directory: {args.ensemble_dir}")
        logger.info(f"Ensemble type: {args.ensemble_type}")
        logger.info(f"Forecast cycle: {args.forecast_cycle} days")
        logger.info(f"Batch size per GPU: {args.batch_size} members")
        logger.info(f"{'='*60}\n")
    
    # Get ensemble files
    member_files = get_ensemble_files(args.ensemble_dir, args.ensemble_type)
    
    if rank == 0:
        logger.info(f"Found {len(member_files)} ensemble members")
    
    # Distribute members across GPUs
    members_per_gpu = args.members_per_gpu if args.members_per_gpu else len(member_files) // world_size + 1
    start_idx = rank * members_per_gpu
    end_idx = min(start_idx + members_per_gpu, len(member_files))
    
    my_members = member_files[start_idx:end_idx]
    
    logger.info(f"[Task {rank}] Assigned {len(my_members)} members (indices {start_idx} to {end_idx-1})")
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(DEFAULT_OUTPUT_LOCATION, args.ensemble_type)
    
    # Process assigned members in batches
    batch_size = args.batch_size
    num_batches = (len(my_members) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(my_members))
        batch_members = my_members[batch_start:batch_end]
        
        logger.info(f"\n[Task {rank}] Processing batch {batch_idx + 1}/{num_batches} ({len(batch_members)} members)")
        
        try:
            create_forecast_batch(
                member_files=batch_members,
                forecast_cycle=args.forecast_cycle,
                output_path=output_dir,
                device=device,
                rank=rank
            )
        except Exception as e:
            logger.error(f"[GPU {rank}] Error processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Try processing members individually as fallback
            logger.info(f"[GPU {rank}] Falling back to individual processing for this batch...")
            for member_file in batch_members:
                try:
                    create_forecast(
                        rdata_path=member_file,
                        forecast_cycle=args.forecast_cycle,
                        output_path=output_dir,
                        device=device,
                        rank=rank
                    )
                except Exception as e2:
                    logger.error(f"[GPU {rank}] Error processing {member_file.name}: {e2}")
                    continue
        
        # Clear cache between batches
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final status
    logger.info(f"[Task {rank}] Completed all assigned batches")
    
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"All ensemble forecasts completed!")
        logger.info(f"Output saved in: {output_dir}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
