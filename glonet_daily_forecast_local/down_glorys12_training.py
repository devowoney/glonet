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

def get_data(start_date : str, 
             end_date : str, 
             depth : int,
             fn) -> Dataset :
    
    id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
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
        start_datetime=start_date,
        end_datetime=end_date,
    )
    
    print(f"Data for depth {depth}:\n", data)
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


def glo_in1(start, end):
    inp = get_data(start, end, 0, f"{MODEL_LOCATION}/xe_weights14/L0.nc")
    print(inp)
    return inp

def glo_in2(start, end):
    depth_list = [50, 100, 150, 222, 318, 380, 450, 540, 640, 763]
    inp = []
    for i in depth_list :
        inp.append(get_data(start, end, i, f"{MODEL_LOCATION}/xe_weights14/L{i}.nc"))
    inp = concat(inp, dim="depth")
    return inp


def glo_in3(start, end):
    depth_list = [902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274]
    inp = []
    for i in depth_list :
        inp.append(get_data(start, end, i, f"{MODEL_LOCATION}/xe_weights14/L{i}.nc"))
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


def create_depth_data(start: date, 
                      end : date, 
                      glo_in, 
                      depth: int):
    dd = glo_in(start, end)
    
    return create_data(dd, depth)


def create_init_states_data(start_date : str, 
                            end_date : str,
                            output_path : str = None) :
    
    function_map = {
        "1" : glo_in1,
        "2" : glo_in2,
        "3" : glo_in3
    }
    
    # Write output file
    if output_path :
        out_location = output_path
    else :
        out_location = DEFAULT_OUTPUT_LOCATION + f"/glorys12_{start_date}_to_{end_date}_init_states"
    
    os.makedirs(out_location, exist_ok=True)
    
    # Collect all datasets
    datasets = []
    for i in ["1", "2", "3"] :
        dataset = create_depth_data(start_date, end_date, function_map[i], int(i) - 1)
        datasets.append(dataset)
        del dataset
    
    # Concatenate along 'ch' dimension
    combined_dataset = concat(datasets, dim="ch")
    del datasets
    gc.collect()
    
    # Reassign the `ch` dimension to ensure it is unique and sequential
    combined_dataset = combined_dataset.assign_coords(
        ch=("ch", range(combined_dataset.sizes["ch"]))
    )

    # Write the concatenated dataset to a single NetCDF file
    output_file = f"{out_location}/combined_input.nc"
    combined_dataset.to_netcdf(output_file)
    
    print(f"Copernicus Marine data is completely downloaded and concatenated in < {output_file} >")
    
    return combined_dataset   

# Parseargs setting
def parse_args () :
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine product data."
    )
    parser.add_argument("--start_date", "-s",
                        dest = "start_date",
                        type = str, 
                        required = True, 
                        help = "Start date in YYYY-MM-DD format.")
    
    parser.add_argument("--end_date", "-e", 
                        dest = "end_date",
                        type = str, 
                        required = True, 
                        help = "End date in YYYY-MM-DD format.")
    
    parser.add_argument("--out_path", "-o",
                        dest = "out_path",
                        type = Path,
                        required = False,
                        help = "Output file path.")
    
    return parser.parse_args()

def main() :
    
    args = parse_args()

    create_init_states_data(start_date= args.start_date, 
                            end_date= args.end_date,
                            output_path= args.out_path)

if __name__ == "__main__":
    main()
