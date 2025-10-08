import copernicusmarine
import argparse
from pathlib import Path
import xarray as xr

def get_data(start_date : str, 
             end_date : str, 
             depth_list : list[float]) -> list[xr.Dataset] :
    
    id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
    var_list = ['zos', 'uo', 'vo', 'so', 'thetao']
    depth_list = depth_list

    ds = []
    for depth in depth_list:
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
    return ds

def parse_args() :
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description="Download Copernicus Marine GLORYS12 data.")

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

    parser.add_argument("--depth", "-d",
                        dest = "depth",
                        type = list, 
                        nargs = '+', 
                        required = False, 
                        help = "List of depths to retrieve data for.")

    parser.add_argument("--out_path", "-o",
                        dest = "out_path",
                        type = Path,
                        required = False,
                        help = "Output file path.")

    return parser.parse_args()

def main() :
    
    args = parse_args()
    if args.depth is None:
        args.depth = [0.5,  50, 100, 150, 222, 318, 380, 450, 540, 640, 763, 
                      902, 1245, 1684, 2225, 3220, 3597, 3992, 4405, 4833, 5274]
    glorys12 = get_data(args.start_date, args.end_date, args.depth)
    dataset = xr.concat(glorys12, dim="depth")
    
    if args.out_path is not None :
        out_file = dataset.to_netcdf(args.out_path / "glorys12_test_data.nc")
    else:
        print("No output path specified. Data will be saved in current directory.")
        out_file = dataset.to_netcdf("glorys12_test_data.nc")

if __name__ == "__main__" :
    main()