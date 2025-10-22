from datetime import datetime, date
from xarray import Dataset, concat, merge, open_dataset
from datetime import timedelta
import copernicusmarine
import numpy
import gc
from xesmf import Regridder
from s3_upload import save_bytes_to_s3
from model import synchronize_model_locally


def get_data(date, depth, fn):
    start_datetime = str(date - timedelta(days=1))
    end_datetime = str(date)
    if depth == 0:
        print("yes")
        mlist = [
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        ]
        mvar = [["zos"], ["uo", "vo"], ["so"], ["thetao"]]
    else:
        print("no")
        mlist = [
            "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        ]
        mvar = [["uo", "vo"], ["so"], ["thetao"]]

    ds = []
    for i in range(0, len(mlist)):
        print(mvar[i])
        data = copernicusmarine.open_dataset(
            dataset_id=mlist[i],
            variables=mvar[i],
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


def glo_in1(model_dir: str, date):
    inp = get_data(date, 0, f"{model_dir}/xe_weights14/L0.nc")
    print(inp)
    return inp


def glo_in2(model_dir: str, date):
    inp = []
    inp.append(get_data(date, 50, f"{model_dir}/xe_weights14/L50.nc"))
    inp.append(get_data(date, 100, f"{model_dir}/xe_weights14/L100.nc"))
    inp.append(get_data(date, 150, f"{model_dir}/xe_weights14/L150.nc"))
    inp.append(get_data(date, 222, f"{model_dir}/xe_weights14/L222.nc"))
    inp.append(get_data(date, 318, f"{model_dir}/xe_weights14/L318.nc"))
    inp.append(get_data(date, 380, f"{model_dir}/xe_weights14/L380.nc"))
    inp.append(get_data(date, 450, f"{model_dir}/xe_weights14/L450.nc"))
    inp.append(get_data(date, 540, f"{model_dir}/xe_weights14/L540.nc"))
    inp.append(get_data(date, 640, f"{model_dir}/xe_weights14/L640.nc"))
    inp.append(get_data(date, 763, f"{model_dir}/xe_weights14/L763.nc"))
    inp = concat(inp, dim="depth")
    return inp


def glo_in3(model_dir: str, date):
    inp = []
    inp.append(get_data(date, 902, f"{model_dir}/xe_weights14/L902.nc"))
    inp.append(get_data(date, 1245, f"{model_dir}/xe_weights14/L1245.nc"))
    inp.append(get_data(date, 1684, f"{model_dir}/xe_weights14/L1684.nc"))
    inp.append(get_data(date, 2225, f"{model_dir}/xe_weights14/L2225.nc"))
    inp.append(get_data(date, 3220, f"{model_dir}/xe_weights14/L3220.nc"))
    inp.append(get_data(date, 3597, f"{model_dir}/xe_weights14/L3597.nc"))
    inp.append(get_data(date, 3992, f"{model_dir}/xe_weights14/L3992.nc"))
    inp.append(get_data(date, 4405, f"{model_dir}/xe_weights14/L4405.nc"))
    inp.append(get_data(date, 4405, f"{model_dir}/xe_weights14/L4405.nc"))
    inp.append(get_data(date, 5274, f"{model_dir}/xe_weights14/L5274.nc"))
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


def create_depth_data(date: date, glo_in, depth: int, model_dir: str):
    dd = glo_in(model_dir, date)
    return create_data(dd, depth)


def create_data_if_needed(
    bucket_name: str,
    forecast_directory_url: str,
    date: date,
    in_layer: str,
    glo_in,
    depth: int,
) -> str:
    init_netcdf_file_url = f"{forecast_directory_url}/inits/{in_layer}.nc"
    try:
        open_dataset(f"{init_netcdf_file_url}#mode=bytes", engine="netcdf4")
        print(f"Initial file already exists: {init_netcdf_file_url}")
    except Exception:
        print(f"Initial file does not exist: {init_netcdf_file_url}")
        local_dir = "/tmp/glonet"
        synchronize_model_locally(local_dir)
        dataset = create_depth_data(date, glo_in, depth, model_dir=local_dir)
        file_key = f"{forecast_directory_url.partition(bucket_name +'/')[2]}/inits/{in_layer}.nc"
        object_bytes = dataset.to_netcdf()
        save_bytes_to_s3(
            bucket_name=bucket_name,
            object_bytes=object_bytes,
            object_key=file_key,
        )
    return init_netcdf_file_url


def generate_initial_data(bucket_name, forecast_netcdf_file_url: str):
    forecast_directory_url = forecast_netcdf_file_url.rpartition("/")[0]
    day_string = forecast_directory_url.rpartition("/")[2]

    date = datetime.strptime(day_string, "%Y-%m-%d").date()
    in1_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in1", glo_in1, 0
    )
    in2_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in2", glo_in2, 1
    )
    in3_url = create_data_if_needed(
        bucket_name, forecast_directory_url, date, "in3", glo_in3, 2
    )
    return in1_url, in2_url, in3_url
