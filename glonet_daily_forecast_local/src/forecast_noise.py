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

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
INPUT_LOCATION = "/Odyssey/public/glonet"
user = os.environ.get("USER")
DEFAULT_OUTPUT_LOCATION = f"/Odyssey/private/{user}/glonet/output"


#####
## Forecast
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
    # d=d.assign(velo=np.sqrt(d['uo']*d['uo']+d['vo']*d['vo']))
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
    # d=d.assign(velo=np.sqrt(d['uo']*d['uo']+d['vo']*d['vo']))
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
    # d=d.assign(velo=np.sqrt(d['uo']*d['uo']+d['vo']*d['vo']))
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
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["contact"] = "glonet@mercator-ocean.eu"
    ds.attrs["institution"] = "Mercator Ocean International"
    ds.attrs["source"] = "MOI GLONET"
    ds.attrs["title"] = (
        "daily mean fields from GLONET 1/4 degree resolution Forecast updated Daily"
    )
    ds.attrs["references"] = "www.edito.eu"

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

def aforecast(d, date, cycle : int):
    from utility import get_denormalizer1, get_normalizer1

    denormalizer = get_denormalizer1(MODEL_LOCATION)
    normalizer = get_normalizer1(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection
    for i in range(1, int((cycle + 1) / 2) + 1):
        print(i)
        model_inf = torch.jit.load(
            MODEL_LOCATION + "/" + "glonet_p1.pt"
        )  # or "scripted_model.pt"
        model_inf = model_inf.to("cuda:0")
        with torch.no_grad():
            model_inf.eval()

        with torch.no_grad():
            with autocast(device_type="cuda:0"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()
    del (
        vin,
        mask,
    )
    gc.collect()
    return datasets

def aforecast2(d, date, cycle):
    from utility import get_denormalizer2, get_normalizer2

    denormalizer = get_denormalizer2(MODEL_LOCATION)
    normalizer = get_normalizer2(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection

    for i in range(1, int((cycle + 1) / 2) + 1):
        print(i)
        model_inf = torch.jit.load(
            MODEL_LOCATION + "/" + "glonet_p2.pt"
        )  # or "scripted_model.pt"
        model_inf = model_inf.to("cuda:0")
        with torch.no_grad():
            model_inf.eval()

        with torch.no_grad():
            with autocast(device_type="cuda:0"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc2(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()
    del (
        vin,
        mask,
    )
    gc.collect()
    return datasets


def aforecast3(d, date, cycle):
    from utility import get_denormalizer3, get_normalizer3

    denormalizer = get_denormalizer3(MODEL_LOCATION)
    normalizer = get_normalizer3(MODEL_LOCATION)
    nan_mask = np.isnan(d.variables["data"][1])
    nan_mask = np.where(nan_mask, 0, 1)
    mask = torch.tensor(nan_mask, dtype=torch.float32)
    data = np.nan_to_num(d.data.data, copy=False)
    vin = torch.tensor(data, dtype=torch.float32)
    mask = mask.cpu().detach().unsqueeze(0)
    vin = normalizer(vin)

    vin = vin.cpu().detach().unsqueeze(0)
    vin = vin.contiguous()
    datasets = []
    del data, nan_mask
    gc.collect()  # Force garbage collection

    for i in range(1, int((cycle + 1) / 2) + 1):
        print(i)
        model_inf = torch.jit.load(
            MODEL_LOCATION + "/" + "glonet_p3.pt"
        )  # or "scripted_model.pt"
        model_inf = model_inf.to("cuda:0")
        with torch.no_grad():
            model_inf.eval()

        with torch.no_grad():
            with autocast(device_type="cuda:0"):
                vin = vin * mask.cpu()
                outvar = model_inf(vin.to("cuda:0"))
                outvar = outvar.detach().cpu()

        del vin
        gc.collect()

        d = make_nc3(outvar, denormalizer, date, i)
        datasets.append(d)
        vin = outvar
        vin = vin.cpu().detach()
        del outvar, model_inf
        gc.collect()
    del (
        vin,
        mask,
    )
    gc.collect()
    return datasets
    
def addRandomNoise(input_ds : xr.Dataset,
                   seed : int = 42) -> xr.Dataset :
    """_summary_

    Sigma value is calulated automatically by the mean of each channel. \n
    Each mean value is divided by 30 to get the standard deviation of the Gaussian noise.
    Max error (3 * sigma) is about 10% of the mean value.
    
    Args:
        input_ds (xr.Dataset): Input Dataset to which noise will be added.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        xr.Dataset: Dataset with added Gaussian noise.
    """

    # Chunk input Dataset
    input_ds = input_ds.chunk(200)
    
    # initialize random noise generater
    rng = np.random.default_rng(seed)

    # Put noise by channel, preserving the original variable name and dataset structure
    arr = input_ds[list(input_ds.data_vars)[0]]
    noised_arrs = []
    
    for i in range(arr.sizes['ch']):
        ch_arr = arr.isel(ch=i)
        mean = ch_arr.mean().compute().item()
        noise = xr.DataArray(
            rng.normal(loc=0, scale=abs(float(mean/30)), size=ch_arr.shape),
            dims=ch_arr.dims,
            coords=ch_arr.coords
        )
        noised_arrs.append((ch_arr + noise).compute())
        
    noised = xr.concat(noised_arrs, dim='ch')
    noised = noised.transpose('time', 'ch', 'lat', 'lon')
    
    # Return dataset with variable name 'data'
    return xr.Dataset({'data': noised})

def create_forecast(init_dir : Path,
                    forecast_cycle : int = None, 
                    output_path : Path = None) -> xr.Dataset :
    # Extract string date
    init_date = str(init_dir).split("_init_", 1)[0].rsplit("/", 1)[1]
    
    # Detect wether the forecast is from GLORYS12 or GLONET forecast states.
    if str(init_dir).split("_init_", 1)[1].split("_")[0] == "from" :
        is_from_glonet_out = True
    else :
        is_from_glonet_out = False
        
    date = datetime.strptime(init_date, "%Y-%m-%d").date()

    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date)
    print(
        f"Creating {init_date} forecast from {start_datetime} to {end_datetime}..."
    )

    start_timed = time.time()
    rdata1 = xr.open_dataset(f"{init_dir}/input1.nc")
    rdata2 = xr.open_dataset(f"{init_dir}/input2.nc")
    rdata3 = xr.open_dataset(f"{init_dir}/input3.nc")
    zicdata1 = addRandomNoise(rdata1)
    zicdata2 = addRandomNoise(rdata2)
    zicdata3 = addRandomNoise(rdata3)
    end_timed = time.time()
    execution_timed = end_timed - start_timed
    start_time = time.time()
    if forecast_cycle :
        forecast_cycle = forecast_cycle
    else : 
        forecast_cycle = 7
    
    ds1 = aforecast(zicdata1, date - timedelta(days=1), cycle=forecast_cycle)
    del rdata1
    gc.collect()
    ds2 = aforecast2(zicdata2, date - timedelta(days=1), cycle=forecast_cycle)
    del rdata2
    gc.collect()
    ds3 = aforecast3(zicdata3, date - timedelta(days=1), cycle=forecast_cycle)
    del rdata3
    gc.collect()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time to get-process data: {execution_timed:.4f} seconds")
    print(f"Time taken by for 10 days on cpu: {execution_time:.4f} seconds")
    combined1 = xr.concat(ds1, dim="time")
    combined2 = xr.concat(ds2, dim="time")
    combined3 = xr.concat(ds3, dim="time")
    del ds1, ds2, ds3
    gc.collect()

    combined4 = xr.concat([combined1, combined2, combined3], dim="depth")
    combined4["zos"] = combined4.zos.isel(depth=0)
    combined4 = add_metadata(combined4, date)

    # Free memory
    del combined1, combined2, combined3
    gc.collect()

    # Write output NetCDF file
    if output_path :
        out_path = output_path
    else : 
        out_path = DEFAULT_OUTPUT_LOCATION
         
    os.makedirs(out_path, exist_ok=True)
    if not is_from_glonet_out :
        combined4.to_netcdf(f"{out_path}/forecast_{forecast_cycle}days_from_{init_date}_noise.nc")
    else :
        combined4.to_netcdf(f"{out_path}/repeated_forecast_{forecast_cycle}days_from_{init_date}_noise.nc")

    print(f"Forecast by GLONET completed : output saved in < {out_path} >")
    
    return combined4

def parse_args ():
    parser = argparse.ArgumentParser(
        description="GLONET forecast - generate forecast ocean states with [Normal Gaussian Random Noise]."
    )
    
    parser.add_argument(
        dest="input_path",
        type=Path,
        help="Input directory which has three initial input files."
                "<input1.nc, input2.nc, input3.nc> must be concluded in input directory."
    )
    
    parser.add_argument(
        "-c", "--cycle",
        dest="forecast_cycle",
        type=int,
        required=False,
        help="Define forecast cycle of GLONET. Default cylce is 7 for 7-day forecast."
    )
    
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        type=Path,
        required=False,
        help="Path to save output file. If output is not given by terminal input, the output will be saved in default location"
    )
    
    return parser.parse_args()



def main() :

    args = parse_args()
    
    input_path = args.input_path
    forecast_cycle = args.forecast_cycle
    output_path = args.output_path
    
    create_forecast(init_dir=input_path, 
                    forecast_cycle=forecast_cycle, 
                    output_path=output_path)

if __name__ == "__main__":
    main()