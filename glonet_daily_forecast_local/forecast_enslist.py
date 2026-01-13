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
from typing import List

# Add current directory to path first to import local utility.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(1, '/Odyssey/private/j25lee/glonet/src/glonet')

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
INPUT_LOCATION = "/Odyssey/public/glonet"
user = os.environ.get("USER")
DEFAULT_OUTPUT_LOCATION = f"/Odyssey/private/{user}/glonet/ensemble_output"


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


def create_forecast(rdata_path: Path,
                    forecast_cycle: int,
                    output_path: str,
                    device) -> xr.Dataset:
    """Create forecast for a single ensemble member"""
    
    # Extract member info
    member_name = rdata_path.stem  # e.g., member_000_initial_condition
    
    print(f"Processing {member_name}...")
    
    # Load data
    rdata = xr.open_dataset(rdata_path)
    date = rdata.time.data[1].astype("M8[D]").astype(datetime)
    
    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date + timedelta(days=forecast_cycle))
    
    print(f"Creating forecast from {start_datetime} to {end_datetime}...")

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
    
    print(f"Forecast time: {execution_time:.4f} seconds")
    
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
    
    print(f"Forecast completed: {output_file}")
    
    return combined4


def get_ensemble_files(ensemble_dir: str) -> List[Path]:
    """Get all ensemble member files from directory"""
    ensemble_path = Path(ensemble_dir)
    
    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory not found: {ensemble_path}")
    
    member_files = sorted(ensemble_path.glob("member_*_initial_condition.nc"))
    
    if not member_files:
        raise ValueError(f"No ensemble members found in {ensemble_path}")
    
    return member_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLONET Ensemble Forward - Single GPU Processing"
    )
    
    parser.add_argument(
        "--ensemble-dir",
        type=str,
        required=True,
        help="Path to ensemble directory"
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Warning: No GPU available, using CPU")
    
    print(f"\n{'='*60}")
    print(f"GLONET Ensemble Forecast - Single GPU Processing")
    print(f"{'='*60}")
    print(f"Ensemble directory: {args.ensemble_dir}")
    print(f"Forecast cycle: {args.forecast_cycle} days")
    print(f"{'='*60}\n")
    
    # Get ensemble files
    member_files = get_ensemble_files(args.ensemble_dir)
    print(f"Found {len(member_files)} ensemble members")
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = DEFAULT_OUTPUT_LOCATION
    
    # Process each member sequentially
    for idx, member_file in enumerate(member_files, 1):
        print(f"\n[{idx}/{len(member_files)}] Processing {member_file.name}")
        
        try:
            create_forecast(
                rdata_path=member_file,
                forecast_cycle=args.forecast_cycle,
                output_path=output_dir,
                device=device
            )
        except Exception as e:
            print(f"Error processing {member_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clear cache between members
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*60}")
    print(f"All ensemble forecasts completed!")
    print(f"Output saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
