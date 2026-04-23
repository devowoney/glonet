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
import glob

#####
## Get Initial Condition from Copernicus Marine #
## SLURM Parallel Version by array
#####

MODEL_LOCATION = "/Odyssey/public/glonet/TrainedWeights"
DEFAULT_OUTPUT_LOCATION = "/Odyssey/public/glonet"

# SLURM environment variable
SLURM_CHUNK_INDEX = "SLURM_ARRAY_TASK_ID"


def iter_date_chunks(start_date: str, end_date: str, chunk_days: int):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer")
    if end_dt < start_dt:
        raise ValueError("end_date must be greater than or equal to start_date")

    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


def get_all_chunks(start_date: str, end_date: str, chunk_days: int):
    """Return all chunks as a list instead of generator (useful for SLURM array jobs)."""
    return list(iter_date_chunks(start_date, end_date, chunk_days))


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
            "ch": ("ch", numpy.arange(0, tt.shape[1])),
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


def process_single_chunk(chunk_start: str,
                         chunk_end: str,
                         output_path: str) -> str:
    """Process a single chunk of data."""
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Processing chunk {chunk_start} to {chunk_end}")

    function_map = {
        "1": glo_in1,
        "2": glo_in2,
        "3": glo_in3
    }

    datasets = []
    for i in ["1", "2", "3"]:
        print(f"  - Fetching depth level {i}")
        dataset = create_depth_data(chunk_start, chunk_end, function_map[i], int(i) - 1)
        datasets.append(dataset)
        del dataset

    combined_dataset = concat(datasets, dim="ch")
    del datasets
    gc.collect()

    combined_dataset = combined_dataset.assign_coords(
        ch=("ch", numpy.arange(combined_dataset.sizes["ch"]))
    )

    chunk_file = f"{output_path}/combined_input_{chunk_start}_to_{chunk_end}.nc"
    combined_dataset.to_netcdf(chunk_file)
    print(f"  Saved: {chunk_file}")

    combined_dataset.close()
    del combined_dataset
    gc.collect()
    
    return chunk_file


def is_slurm_environment():
    """Check if running under SLURM."""
    return SLURM_CHUNK_INDEX in os.environ


def process_slurm_chunk(start_date: str,
                        end_date: str,
                        output_path: str,
                        chunk_days: int = 30):
    """Process a single chunk assigned to this SLURM task."""
    chunk_index = int(os.environ.get(SLURM_CHUNK_INDEX, "1")) - 1
    all_chunks = get_all_chunks(start_date, end_date, chunk_days)
    
    if chunk_index >= len(all_chunks):
        print(f"ERROR: Chunk index {chunk_index} out of range. Total chunks: {len(all_chunks)}")
        sys.exit(1)
    
    chunk_start, chunk_end = all_chunks[chunk_index]
    print(f"[Task {chunk_index + 1}/{len(all_chunks)}] {chunk_start} to {chunk_end}")
    
    return process_single_chunk(chunk_start, chunk_end, output_path)


def create_init_states_data(start_date: str,
                            end_date: str,
                            output_path: str = None,
                            chunk_days: int = 30,
                            combine_chunks: bool = False,
                            slurm_mode: bool = False) -> list:
    """Create initial states data from Copernicus Marine."""
    
    if output_path:
        out_location = output_path
    else:
        out_location = DEFAULT_OUTPUT_LOCATION + f"/glorys12_{start_date}_to_{end_date}_init_states"
    
    os.makedirs(out_location, exist_ok=True)
    
    # SLURM mode: process only the assigned chunk
    if slurm_mode or is_slurm_environment():
        chunk_file = process_slurm_chunk(start_date, end_date, out_location, chunk_days)
        return [chunk_file]
    
    # Sequential mode: process all chunks locally
    chunk_files = []
    for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_days):
        chunk_file = process_single_chunk(chunk_start, chunk_end, out_location)
        chunk_files.append(chunk_file)

    if combine_chunks:
        print(f"\nCombining {len(chunk_files)} chunks...")
        datasets_to_combine = [open_dataset(path) for path in chunk_files]
        merged_all = concat(datasets_to_combine, dim="time")
        output_file = f"{out_location}/combined_input.nc"
        merged_all.to_netcdf(output_file)
        for ds in datasets_to_combine:
            ds.close()
        merged_all.close()
        gc.collect()
        print(f"Combined file: {output_file}")

    print(f"\nCompleted: {len(chunk_files)} chunks in {out_location}")
    return chunk_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine data. Auto-detects SLURM environment."
    )
    parser.add_argument("--start_date", "-s",
                        required=True, 
                        help="Start date (YYYY-MM-DD)")
    
    parser.add_argument("--end_date", "-e", 
                        required=True, 
                        help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--out_path", "-o",
                        help="Output directory")

    parser.add_argument("--chunk_days", "-c",
                        type=int,
                        default=30,
                        help="Days per chunk (default: 30)")

    parser.add_argument("--combine_chunks",
                        action="store_true",
                        help="Combine chunks into single file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_path = args.out_path if args.out_path else DEFAULT_OUTPUT_LOCATION + f"/glorys12_{args.start_date}_to_{args.end_date}_init_states"
    
    # Auto-detect SLURM and run accordingly
    create_init_states_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
        chunk_days=args.chunk_days,
        combine_chunks=args.combine_chunks,
        slurm_mode=is_slurm_environment()  # Auto-detect
    )


if __name__ == "__main__":
    main()
