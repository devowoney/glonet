from datetime import datetime, date
from xarray import Dataset, concat, merge, open_dataset
from datetime import timedelta
import copernicusmarine
import numpy
import gc
from xesmf import Regridder
import os
import sys
import argparse
from pathlib import Path

#####
## Get Initial Condition from Copernicus Marine #
#####

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
DEFAULT_OUTPUT_LOCATION = "/Odyssey/public/glonet"

def get_data(date, depth, fn):
    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date)
    
    id = 'cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m'
    if depth == 0:
        print("yes")
        var_list = ['zos', 'uo', 'vo', 'so', 'thetao']
        depth = 0.5
    else:
        print("no")
        var_list = ['uo', 'vo', 'so', 'thetao']

    ds = []

    data = copernicusmarine.open_dataset(
        dataset_id=id,
        variables=var_list,
        minimum_longitude=-180,
        maximum_longitude=180,
        minimum_latitude=-80,
        maximum_latitude=90,
        minimum_depth=depth,
        maximum_depth=depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    
    ds.append(data)
    print("merging..")
    ds = merge(ds)
    print(ds)
    ds_out = Dataset(
        {
            "lat": (
                ["lat"],
                numpy.arange(data.latitude.min(), data.latitude.max(), 1 / 4),
            ),
            "lon": (
                ["lon"],
                numpy.arange(
                    data.longitude.min(), data.longitude.max(), 1 / 4
                ),
            ),
        }
    )

    print("loading regridder")
    regridder = Regridder(
        data, ds_out, "bilinear", weights=fn, reuse_weights=True
    )
    print("regridder ready")
    ds_out = regridder(ds)
    print("done regridding")
    ds_out = ds_out.sel(lat=slice(ds_out.lat[8], ds_out.lat[-1]))
    # print(ds_out)
    del regridder, ds, data
    gc.collect()
    return ds_out


def glo_in1(date):
    inp = get_data(date, 0, f"{MODEL_LOCATION}/xe_weights14/L0.nc")
    print(inp)
    return inp

def glo_in2(date):
    depth_list = [50, 100, 150, 222, 318, 380, 450, 540, 640, 763]
    inp = []
    for i in depth_list :
        inp.append(get_data(date, i, f"{MODEL_LOCATION}/xe_weights14/L{i}.nc"))
    inp = concat(inp, dim="depth")
    return inp


def glo_in3(date):
    depth_list = [902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274]
    inp = []
    for i in depth_list :
        inp.append(get_data(date, i, f"{MODEL_LOCATION}/xe_weights14/L{i}.nc"))
    inp = concat(inp, dim="depth")
    return inp


def create_data(ds_out, depth):
    thetao = ds_out["thetao"].data
    so = ds_out["so"].data
    uo = ds_out["uo"].data
    vo = ds_out["vo"].data
    if depth == 0:
        zos = numpy.expand_dims(ds_out["zos"].data, axis=1)
        tt = numpy.concatenate([zos, thetao, so, uo, vo], axis=1)
    else:
        tt = numpy.concatenate([thetao, so, uo, vo], axis=1)

    lat = ds_out.lat.data
    lon = ds_out.lon.data
    time = ds_out.time.data

    bb = Dataset(
        {
            "data": (("time", "ch", "lat", "lon"), tt),
        },
        coords={
            "time": ("time", time),
            "ch": ("ch", range(0, tt.shape[1])),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    return bb


def create_depth_data(date: date, glo_in, depth: int):
    dd = glo_in(date)
    
    return create_data(dd, depth)


def create_init_states_data(target_date : str, 
                            output_path : str = None) :
    date = datetime.strptime(target_date, "%Y-%m-%d")
    
    function_map = {
        "1" : glo_in1,
        "2" : glo_in2,
        "3" : glo_in3
    }
    
    # Write output file
    if output_path :
        out_location = output_path
    else :
        out_location = DEFAULT_OUTPUT_LOCATION + f"/{target_date}_climatology"
    
    os.makedirs(out_location, exist_ok=True)
    
    for i in ["1", "2", "3"] :
        dataset = create_depth_data(date, function_map[i], int(i) - 1)
        dataset.to_netcdf(f"{out_location}/mean{int(i)}.nc")
        
    print(f"Copernicus Marine data is completely downloaded in < {out_location} >")
    
    return dataset   

# Parseargs setting
def parse_args () :
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine product data."
    )
    parser.add_argument(
        dest="input_date",
        type=str,
        help="Input : a date to download copernicus marine data. Format : yyyy-mm-dd"
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

    input_date = args.input_date
    output_path = args.output_path
    create_init_states_data(target_date=input_date, 
                            output_path=output_path)

if __name__ == "__main__":
    main()