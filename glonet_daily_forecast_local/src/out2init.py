import xarray as xr
import numpy as np
import os

import argparse
import sys
from pathlib import Path



# Reshape Forecast Dataset to GLONET input (3 files) Dataset shape. 
# i.e. flatten depth dimension and passe it to channel with variables.

# Define Final type variable.
DEFAULT_OUTPUT_LOCATION = "/Odyssey/public/glonet"

def reshapeDataset(in_ds : xr.Dataset) -> xr.Dataset:
        
    out_ds = (in_ds.to_array(dim = "variable")                              # Convert to xarray DatArray
                    .stack(ch=("variable", "depth"))                        # Merge two dimensions one dimension "ch"
                    .reset_index("ch", drop=True)                           # Clear void coordinates 
                    .assign_coords(ch=np.arange(len(in_ds.variables) * in_ds.sizes["depth"])) # define values of ch 
                    .transpose("time", "ch", "latitude", "longitude")       # Reshape in order
                    .rename({"latitude" : "lat", "longitude" : "lon"})      # Rename into input Dataset config
                    .to_dataset)                                            # Reconvert to Dataset
        
    return out_ds

# Divide forcast NetCDF file in to 3 files by states depth
def makeThreeInput(input_nc : str, 
                    cycle : int,
                    output_path : str = None) -> list[xr.Dataset] :
    r"""
    input_nc is the forecast path which is the issue of glonet.
    
    Here, cycle stands for the extraction of the forecast data in interest.
     
    It is NOT same as the cycle of the forecast python script. And of course, this cycle should be smaller than forecast cycle
    """
    
    # Extract date for output name
    forecast_date = xr.open_dataset(input_nc)["time"].dt.date.values[cycle - 1]
    
    # Read netCDF file and pick only two last timeset
    last_two_day_forecast = xr.open_dataset(input_nc).isel(time = [cycle-2, cycle-1])
    
    # Slice datatset by depth first 
    surface = last_two_day_forecast.isel(depth = 0, missing_dims = "ignore")
    deep = last_two_day_forecast.sel(depth = slice(40, 800))
    deeper = last_two_day_forecast.sel(depth = slice(900, 5500))
    
    # Make a dictionary for iterattion
    ds_map = {
            "1" : surface,
            "2" : deep,
            "3" : deeper
    }
    
    # Write output NetCDF file.
    if output_path :
        out_path = output_path
    else : 
        out_path = DEFAULT_OUTPUT_LOCATION
    
    os.makedirs(out_path, exist_ok=True)
    
    # initialize a list
    in123 = []
        
    for i in ds_map :
        in123.append(i)
        i.to_netcdf(f"{out_path}/{forecast_date}_autoregress_init_states")
        
    return in123

# Parseargs setting
def parse_args () :
    parser = argparse.ArgumentParser(
        description="Process some data and write to an output path."
    )
    parser.add_argument(
        dest = "input_path",
        type = Path,
        required = True,
        help = "Input of the function : NetCDF format file."
    )
    parser.add_argument(
        dest = "autoregress_cycle",
        type = str,
        required = True,
        help = """Autoregression cyle to extract last day forecast. 
        It can be different with forecast cycle. But make sure that the cycle should be smaller than the forecast cycle.
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        dest = "output_path",
        type = Path,
        required = False,
        help = "Path to save output file. If output is not given by terminal input, the output will be saved in default location"
    )
    
    return parser.parse_args()
        


def main() :

    args = parse_args()
    
    input_file = args.input_path
    autoregress_cycle = args.autoregress_cycle
    output_path = args.output_path
    
    output_list = makeThreeInput(input_nc=input_file,
                                 cycle=autoregress_cycle,
                                 output_path=output_path)
    
if __name__ == "__main__" :
    main()
