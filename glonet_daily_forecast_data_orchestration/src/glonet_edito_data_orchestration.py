from datetime import timedelta
from os import environ
from typing import Any

from xarray import Dataset, open_dataset, Variable
from functools import partial
from numpy import datetime64
from requests import Response, post, put, delete, get
from datetime import datetime
import json
import base64
from dataclasses import dataclass
from urllib import request

from generate_thumbnails import generate_thumbnail
from minio import Minio
from s3_upload import delete_object, list_objects, save_bytes_to_s3

from get_inits import generate_initial_data
from glonet_forecast import generate_forecast_file

EDITO_DATA_API_URL = "https://api.dive.edito.eu/data"
EDITO_DATA_CATALOG_URL = f"{EDITO_DATA_API_URL}/catalogs"
EDITO_DATA_COLLECTION_URL = f"{EDITO_DATA_API_URL}/collections"

GLONET_PAPER = "https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1524960"

NUMBER_OF_FORECAST_INITS_TO_KEEP = 30
NUMBER_OF_FORECASTS_TO_KEEP = 15
NUMBER_OF_DAYS_TO_FORECAST = 10

GLONET_BUCKET = "project-glonet"


@dataclass(frozen=True)
class ForecastInformation:
    netcdf_file_url: str
    forecast_initial_day: str
    forecast_start_day: str
    forecast_end_day: str
    reference_forecast_file: bool


def _datetime_to_string(datetime64: datetime64) -> str:
    return datetime.fromisoformat(str(datetime64)).strftime("%Y-%m-%d %H:%M")


def _datetime_to_day(datetime64: datetime64 | datetime) -> str:
    return datetime.fromisoformat(str(datetime64)).strftime("%Y-%m-%d")


def _convert_snake_to_title(snake_str):
    return snake_str.replace("_", " ").capitalize()


def _generate_notebook_json(netcdf_file_url: str) -> str:
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "!conda install -y xarray numpy netcdf4 matplotlib"
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import xarray as xr\n",
                        "import matplotlib.pyplot as plt\n",
                        "import numpy as np\n",
                        "\n",
                        f"ds=xr.open_dataset('{netcdf_file_url}#mode=bytes')",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "fig, ax = plt.subplots(4, 1, figsize=(15, 25))\n",
                        "lead=3\n",
                        "depth=0\n",
                        "\n",
                        "ds.vo[lead,depth].plot(ax=ax[0])\n",
                        'ds.thetao[lead,depth].plot(ax=ax[1], vmin=-3, vmax=33, cmap="jet")\n',
                        "ds.so[lead,depth].plot(ax=ax[2], vmin=0)\n",
                        "ds.zos[lead].plot(ax=ax[3])\n",
                        "plt.tight_layout()",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3 (ipykernel)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.12.6",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def _generate_notebook_data_uri(netcdf_file_url: str) -> str:
    encoded_content = base64.b64encode(
        _generate_notebook_json(netcdf_file_url).encode("utf-8")
    ).decode("utf-8")
    data_uri = (
        f"data:application/x-ipynb+json;charset=utf-8;base64,{encoded_content}"
    )
    return data_uri


@dataclass(frozen=True)
class ForecastMetadata:
    forecast_dataset_metadata: Dataset
    thumbnail_urls: dict[str, str]


def _make_feature(
    forecast_dataset: Dataset,
    variable_name: str,
    netcdf_file_url: str,
    thumbnail_url: str,
    reference_forecast_file: bool,
):
    variable = forecast_dataset.variables[variable_name]
    glonet_version = "0.1.0"
    glonet_citation = "A. El Aouni et al, 2024, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM"
    start_datetime: datetime64 = forecast_dataset["time"][0].values
    end_datetime: datetime64 = forecast_dataset["time"][-1].values
    feature_id = f"glonet_1_4_daily_forecast-{variable.attrs['standard_name']}-{_datetime_to_day(start_datetime)}-{_datetime_to_day(end_datetime)}"
    latitude_min = float(forecast_dataset.latitude.valid_min)
    latitude_max = float(forecast_dataset.latitude.valid_max)
    longitude_min = float(forecast_dataset.longitude.valid_min)
    longitude_max = float(forecast_dataset.longitude.valid_max)
    thumbnail_asset = {
        "thumbnail": {
            "href": thumbnail_url,
            "title": "Thumbnail",
            "type": "image/jpeg",
            "roles": ["thumbnail"],
        },
    }
    netcdf_file_asset = {
        "netcdf": {
            "href": netcdf_file_url,
            "title": "NetCDF",
            "roles": ["data"],
            "type": None,
            "description": 'NetCDF file"',
        },
    }
    assets = (
        (thumbnail_asset | netcdf_file_asset)
        if reference_forecast_file
        else thumbnail_asset
    )
    links = (
        [
            {
                "rel": "example",
                "href": _generate_notebook_data_uri(netcdf_file_url),
                "title": "Jupyter Notebook to open the dataset and plot result",
                "type": "application/x-ipynb+json",
                "example:language": "Multiple",
                "example:container": True,
            }
        ]
        if reference_forecast_file
        else []
    )
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        longitude_min,
                        latitude_min,
                    ],
                    [
                        longitude_max,
                        latitude_min,
                    ],
                    [
                        longitude_max,
                        latitude_max,
                    ],
                    [
                        longitude_min,
                        latitude_max,
                    ],
                    [
                        longitude_min,
                        latitude_min,
                    ],
                ]
            ],
        },
        "properties": {
            "title": f"{_convert_snake_to_title(variable.attrs['standard_name'])} daily forecast of GLONET 1/4 degree resolution, from {_datetime_to_string(start_datetime)} to {_datetime_to_string(end_datetime)}",
            "description": f"{_convert_snake_to_title(variable.attrs['standard_name'])} daily forecast of GLONET 1/4 degree resolution, from {_datetime_to_string(start_datetime)} to {_datetime_to_string(end_datetime)}, produced by GLONET 1/4 version {glonet_version}. More information in [AGU 2024 GLONET paper]({GLONET_PAPER})",
            "start_datetime": str(start_datetime),
            "end_datetime": str(end_datetime),
            "institution": "Mercator Ocean International",
            "source": "EDITO-ModelLab",
            "sci:publications": [{"citation": glonet_citation}],
            "cube:dimensions": {
                "latitude": {
                    "extent": [
                        latitude_min,
                        latitude_max,
                    ],
                    "type": "spatial",
                    "axis": "y",
                    "step": forecast_dataset.latitude.step,
                    "units": "degrees_north",
                    "values": list(
                        map(float, forecast_dataset.latitude.values)
                    ),
                },
                "longitude": {
                    "extent": [
                        longitude_min,
                        longitude_max,
                    ],
                    "type": "spatial",
                    "axis": "x",
                    "step": forecast_dataset.longitude.step,
                    "units": "degrees_east",
                    "values": list(
                        map(float, forecast_dataset.longitude.values)
                    ),
                },
                "depth": {
                    "values": list(map(float, forecast_dataset.depth.values)),
                    "type": "spatial",
                    "axis": "z",
                    "step": None,
                    "unit": "meter",
                },
                "time": {
                    "extent": [
                        longitude_min,
                        longitude_max,
                    ],
                    "type": "temporal",
                    "step": "P1D",
                    "values": list(map(str, forecast_dataset.time.values)),
                },
            },
            "cube:variables": {
                variable_name: {
                    "id": variable_name,
                    "standardName": variable.attrs["standard_name"],
                    "unit": variable.attrs["units"],
                    "dimensions": ["latitude", "longitude", "elevation"],
                }
            },
            "cf:parameter": [
                {
                    "name": variable.attrs["standard_name"],
                    "unit": variable.attrs["units"],
                }
            ],
        },
        "assets": assets,
        "links": links,
        "stac_version": "1.0.0",
        "bbox": [
            longitude_min,
            latitude_min,
            longitude_max,
            latitude_max,
        ],
    }


def _delete_edito(resource_url: str) -> Response:
    result: Response = delete(
        resource_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {environ['EDITO_ACCESS_TOKEN']}",
        },
    )
    if not result.ok:
        print(f"Error while creating then deleting resource {resource_url}")
        print(result.text)
    else:
        print(f"Resource {resource_url} deleted")
    return result


def _put_post_edito(url: str, resource) -> Response:
    resource_url = f'{url}/{resource["id"]}'
    put_result: Response = put(
        resource_url,
        json=resource,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {environ['EDITO_ACCESS_TOKEN']}",
        },
    )
    if not put_result.ok:
        post_result: Response = post(
            url,
            json=resource,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {environ['EDITO_ACCESS_TOKEN']}",
            },
        )
        if not post_result.ok:
            print(
                f"Error while updating then creating resource {resource_url}"
            )
            print(post_result.text)
        else:
            print(f"Resource {resource_url} created")
        return post_result
    print(f"Resource {resource_url} updated")
    return put_result


def _make_forecast_dataset_without_data(forecast_dataset: Dataset) -> Dataset:
    return Dataset(
        {
            var: Variable(
                dims=[], data=None, attrs=forecast_dataset[var].attrs
            )
            for var in forecast_dataset.data_vars
        },
        coords=forecast_dataset.coords,
        attrs=forecast_dataset.attrs,
    )


def _generate_forecast_dataset_without_data(
    forecast_information: ForecastInformation,
):
    try:
        print(
            f"Trying to avoid generating {forecast_information.forecast_initial_day} forecast dataset without data..."
        )
        _get_forecast_dataset_without_data(
            forecast_information.netcdf_file_url
        )
        print(
            f"Successfully got {forecast_information.forecast_initial_day} forecast dataset without data"
        )
    except Exception:
        forecast_netcdf_file_without_data_url = (
            _forecast_netcdf_file_without_data_url(
                forecast_information.netcdf_file_url
            )
        )
        print(
            f"Forecast dataset without data does not exist, generating {forecast_netcdf_file_without_data_url}..."
        )
        forecast_dataset = _get_forecast_dataset(forecast_information)
        forecast_dataset_without_data = _make_forecast_dataset_without_data(
            forecast_dataset
        )
        file_key = f"{forecast_netcdf_file_without_data_url.partition(GLONET_BUCKET + '/')[2]}"
        object_bytes = forecast_dataset_without_data.to_netcdf()
        save_bytes_to_s3(
            bucket_name=GLONET_BUCKET,
            object_bytes=object_bytes,
            object_key=file_key,
        )


def _remove_forecast_dataset_with_data(forecast_netcdf_file_url: str):
    file_key = f"{forecast_netcdf_file_url.partition(GLONET_BUCKET + '/')[2]}"
    delete_object(bucket_name=GLONET_BUCKET, object_key=file_key)


def _remove_forecast_dataset_without_data(forecast_netcdf_file_url: str):
    forecast_netcdf_file_without_data_url = (
        _forecast_netcdf_file_without_data_url(forecast_netcdf_file_url)
    )
    file_key = f"{forecast_netcdf_file_without_data_url.partition(GLONET_BUCKET + '/')[2]}"
    delete_object(bucket_name=GLONET_BUCKET, object_key=file_key)


def _keep_forecast_dataset_file_with_data(
    forecast_information: ForecastInformation,
):
    _get_forecast_dataset(forecast_information)
    _remove_forecast_dataset_without_data(forecast_information.netcdf_file_url)


def _keep_forecast_dataset_file_without_data(
    forecast_information: ForecastInformation,
):
    _generate_forecast_dataset_without_data(forecast_information)
    _remove_forecast_dataset_with_data(forecast_information.netcdf_file_url)


def _publish_item(
    forecast_metadata: ForecastMetadata,
    netcdf_file_url: str,
    reference_forecast_file: bool,
    variable_name: str,
) -> str | None:
    variable = forecast_metadata.forecast_dataset_metadata.variables[
        variable_name
    ]
    collection_name = f"climate_forecast-{variable.attrs['standard_name']}"
    feature = _make_feature(
        forecast_metadata.forecast_dataset_metadata,
        variable_name,
        netcdf_file_url,
        forecast_metadata.thumbnail_urls[variable_name],
        reference_forecast_file,
    )
    collection_item_url = (
        f"{EDITO_DATA_COLLECTION_URL}/{collection_name}/items"
    )
    result = _put_post_edito(collection_item_url, feature)
    return f'{collection_item_url}/{feature["id"]}' if result.ok else None


def _publish_glonet_root_catalog():
    print("Publishing GLONET catalog...")
    glonet_catalog = {
        "id": "glonet",
        "stac_version": "1.0.0",
        "type": "Catalog",
        "title": "GLONET",
        "description": f"Catalog gathering all Mercator Ocean international's GLONET AI system assets. GLONET is developed as part of the EDITO-ModelLab project. More information in [AGU 2024 GLONET paper]({GLONET_PAPER})",
        "assets": {
            "thumbnail": {
                "href": "https://minio.dive.edito.eu/project-glonet/public/glonet_thumbnail.png",
                "title": "Thumbnail",
                "type": "image/png",
                "roles": ["thumbnail"],
            },
        },
    }
    glonet_catalog_creation_result = _put_post_edito(
        EDITO_DATA_CATALOG_URL, glonet_catalog
    )
    assert (
        glonet_catalog_creation_result.ok
    ), glonet_catalog_creation_result.text


def _publish_glonet_1_4_daily_forecast_catalog():
    print("Publishing GLONET 1/4 daily forecast catalog...")
    glonet_catalog_url = f"{EDITO_DATA_CATALOG_URL}/glonet"
    daily_forecast_catalog = {
        "id": "glonet_1_4_daily_forecast",
        "stac_version": "1.0.0",
        "type": "Catalog",
        "title": "GLONET 1/4 degree resolution daily forecast",
        "description": "Catalog gathering all assets of GLONET 1/4 degree resolution daily forecast system, updated daily",
    }
    daily_forecast_catalog_creation_result = _put_post_edito(
        glonet_catalog_url, daily_forecast_catalog
    )
    assert (
        daily_forecast_catalog_creation_result.ok
    ), daily_forecast_catalog_creation_result.text


def _publish_forecast_catalog(
    forecast_information: ForecastInformation,
    feature_urls: list[str | None],
):
    print(
        f"Publishing GLONET 1/4 daily forecast catalog for day {forecast_information.forecast_initial_day}..."
    )
    glonet_daily_forecast_catalog_url = (
        f"{EDITO_DATA_CATALOG_URL}/glonet/glonet_1_4_daily_forecast"
    )
    daily_forecast_catalog = {
        "id": f"glonet_1_4_{forecast_information.forecast_initial_day}_forecast",
        "stac_version": "1.0.0",
        "type": "Catalog",
        "title": f"GLONET 1/4 degree resolution {forecast_information.forecast_initial_day} forecast",
        "description": f"Catalog gathering all assets of GLONET 1/4 degree resolution {forecast_information.forecast_initial_day} forecast, from {forecast_information.forecast_start_day} to {forecast_information.forecast_end_day}",
        "links": list(
            map(
                lambda url: {
                    "rel": "item",
                    "href": url,
                    "title": None,
                    "type": "application/json",
                },
                filter(lambda urls: urls is not None, feature_urls),
            )
        ),
    }
    daily_forecast_catalog_creation_result = _put_post_edito(
        glonet_daily_forecast_catalog_url, daily_forecast_catalog
    )
    assert (
        daily_forecast_catalog_creation_result.ok
    ), daily_forecast_catalog_creation_result.text


def _forecast_information_from_forecast_initial_datetime(
    forecast_initial_datetime: datetime,
    reference_forecast_file: bool,
) -> ForecastInformation:
    forecast_initial_day = _datetime_to_day(forecast_initial_datetime)
    forecast_start_day = _datetime_to_day(
        forecast_initial_datetime + timedelta(days=1)
    )
    forecast_end_day = _datetime_to_day(
        forecast_initial_datetime + timedelta(days=NUMBER_OF_DAYS_TO_FORECAST)
    )
    return ForecastInformation(
        netcdf_file_url=f"https://minio.dive.edito.eu/project-glonet/public/glonet_1_4_daily_forecast/{forecast_initial_day}/GLONET_MOI_{forecast_start_day}_{forecast_end_day}.nc",
        forecast_initial_day=forecast_initial_day,
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
        reference_forecast_file=reference_forecast_file,
    )


def _get_forecast_metadata(
    forecast_information: ForecastInformation,
) -> ForecastMetadata:
    thumbnail_urls = _get_forecast_thumbnail_urls(forecast_information)
    forecast_dataset_metadata = _get_forecast_dataset_metadata(
        forecast_information
    )
    return ForecastMetadata(
        forecast_dataset_metadata=forecast_dataset_metadata,
        thumbnail_urls=thumbnail_urls,
    )


def _publish_forecast_catalog_and_features(
    forecast_information: ForecastInformation,
):

    print(
        f"Publishing catalog and features for {forecast_information.forecast_initial_day} forecast from {forecast_information.forecast_start_day} to {forecast_information.forecast_end_day}..."
    )
    try:
        forecast_metadata = _get_forecast_metadata(forecast_information)
        feature_urls = list(
            map(
                partial(
                    _publish_item,
                    forecast_metadata,
                    forecast_information.netcdf_file_url,
                    forecast_information.reference_forecast_file,
                ),
                forecast_scientific_variables(
                    forecast_metadata.forecast_dataset_metadata
                ),
            )
        )
        _publish_forecast_catalog(
            forecast_information,
            feature_urls,
        )
        if not forecast_information.reference_forecast_file:
            _keep_forecast_dataset_file_without_data(forecast_information)
        else:
            _keep_forecast_dataset_file_with_data(forecast_information)
        print(
            f"Successfully published catalog and features for {forecast_information.forecast_initial_day} forecast from {forecast_information.forecast_start_day} to {forecast_information.forecast_end_day}"
        )
    except Exception as exception:
        print(
            f"Could not publish catalog and features for day {forecast_information.forecast_initial_day}: {exception}"
        )
        raise


def _publish_catalogs_and_features(
    forecast_informations_to_resolve: list[ForecastInformation],
):
    list(
        map(
            _publish_forecast_catalog_and_features,
            forecast_informations_to_resolve,
        )
    )


def _get_forecast_initial_datetimes() -> list[datetime]:
    response = get(
        "https://api.dive.edito.eu/data/collections/climate_forecast-northward_sea_water_velocity/items/GLOBAL_ANALYSISFORECAST_PHY_001_024-cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m-202406-default-arco-geo-series-northward_sea_water_velocity-vo"
    )
    assert response.ok, response.text
    feature = response.json()
    last_forecast_datetime = datetime.fromisoformat(
        feature["properties"]["end_datetime"]
    ) - timedelta(days=9)
    print(f"Updated datetime for GLO12 datasets: {last_forecast_datetime}")
    return list(
        last_forecast_datetime - timedelta(days=i)
        for i in range(0, NUMBER_OF_FORECAST_INITS_TO_KEEP)
    )


def _get_forecast_informations_to_resolve() -> list[ForecastInformation]:
    forecast_informations_to_resolve = [
        _forecast_information_from_forecast_initial_datetime(
            forecast_initial_datetime, index < NUMBER_OF_FORECASTS_TO_KEEP
        )
        for index, forecast_initial_datetime in enumerate(
            _get_forecast_initial_datetimes()
        )
    ]
    print("Forecast informations to resolve:")
    list(map(print, forecast_informations_to_resolve))
    return forecast_informations_to_resolve


def _get_forecast_dataset_metadata(
    forecast_information: ForecastInformation,
) -> Dataset:
    try:
        print(
            f"Trying to get {forecast_information.forecast_initial_day} forecast metadata from forecast dataset without data..."
        )
        forecast_dataset_without_data = _get_forecast_dataset_without_data(
            forecast_information.netcdf_file_url
        )
        print(
            f"Got {forecast_information.forecast_initial_day} forecast metadata from forecast dataset without data"
        )
        return forecast_dataset_without_data
    except Exception:
        print(
            f"Failed to get {forecast_information.forecast_initial_day} forecast metadata from forecast dataset without data"
        )
        print(
            f"Trying to get {forecast_information.forecast_initial_day} forecast metadata from forecast dataset..."
        )
        forecast_dataset = _get_forecast_dataset(forecast_information)
        print(
            f"Got {forecast_information.forecast_initial_day} forecast metadata from forecast dataset"
        )
        return forecast_dataset


def _get_forecast_thumbnail_urls(
    forecast_information: ForecastInformation,
) -> dict[str, str]:
    forecast_directory_url = forecast_information.netcdf_file_url.rpartition(
        "/"
    )[0]
    forecast_dataset_metadata = _get_forecast_dataset_metadata(
        forecast_information
    )
    thumbnail_file_urls = {
        str(variable_key): f"{forecast_directory_url}/{variable_key}_ortho.png"
        for variable_key in forecast_scientific_variables(
            forecast_dataset_metadata
        )
    }
    thumbnail_file_url = thumbnail_file_urls["vo"]
    try:
        print(
            f"Trying to get thumbnails for resource {forecast_information.netcdf_file_url}..."
        )
        request.urlopen(thumbnail_file_url)
        print(f"Thumbnail file already exists: {thumbnail_file_url}")
        return thumbnail_file_urls
    except Exception:
        print(f"Thumbnail file does not exist: {thumbnail_file_url}")
        forecast_dataset = _get_forecast_dataset(forecast_information)
        return generate_thumbnail(
            GLONET_BUCKET,
            forecast_information.netcdf_file_url,
            thumbnail_file_urls,
            forecast_dataset,
        )


def forecast_scientific_variables(forecast_dataset: Dataset) -> list[str]:
    return list(map(str, forecast_dataset.data_vars))


def _get_initial_file_urls(
    forecast_information: ForecastInformation,
) -> tuple[str, str, str]:
    print(
        f"Getting initial files for day {forecast_information.forecast_initial_day}"
    )
    return generate_initial_data(
        GLONET_BUCKET, forecast_information.netcdf_file_url
    )


def _forecast_netcdf_file_without_data_url(
    forecast_netcdf_file_url: str,
) -> str:
    return forecast_netcdf_file_url.rpartition(".nc")[0] + "-without-data.nc"


def _get_forecast_dataset_without_data(
    forecast_netcdf_file_url: str,
) -> Dataset:
    forecast_netcdf_file_without_data_url = (
        _forecast_netcdf_file_without_data_url(forecast_netcdf_file_url)
    )
    print(
        f"Getting forecast dataset without data {forecast_netcdf_file_without_data_url}"
    )
    return open_dataset(
        f"{forecast_netcdf_file_without_data_url}#mode=bytes", engine="netcdf4"
    )


def _get_forecast_dataset(
    forecast_information: ForecastInformation,
) -> Dataset:
    initial_file_1_url, initial_file_2_url, initial_file_3_url = (
        _get_initial_file_urls(forecast_information)
    )
    print(
        f"Generating forecast dataset {forecast_information.netcdf_file_url}..."
    )
    generate_forecast_file(
        GLONET_BUCKET,
        forecast_information.netcdf_file_url,
        initial_file_1_url,
        initial_file_2_url,
        initial_file_3_url,
    )
    print(
        f"Successfully generated forecast dataset {forecast_information.netcdf_file_url}, opening it..."
    )
    dataset = open_dataset(
        f"{forecast_information.netcdf_file_url}#mode=bytes", engine="netcdf4"
    )
    print(
        f"Successfully opened forecast dataset {forecast_information.netcdf_file_url}"
    )
    return dataset


def _share_s3_public_directory():

    print(f"Setup {GLONET_BUCKET} bucket public policy...")

    client = Minio(
        endpoint=environ["AWS_S3_ENDPOINT"],
        access_key=environ["AWS_ACCESS_KEY_ID"],
        secret_key=environ["AWS_SECRET_ACCESS_KEY"],
        session_token=environ["AWS_SESSION_TOKEN"],
    )

    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetBucketLocation"],
                "Resource": [f"arn:aws:s3:::{GLONET_BUCKET}"],
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{GLONET_BUCKET}"],
                "Condition": {"StringEquals": {"s3:prefix": ["public"]}},
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{GLONET_BUCKET}/public*"],
            },
        ],
    }
    client.set_bucket_policy(GLONET_BUCKET, json.dumps(bucket_policy))


def _human_readable_size(size_in_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"


def _print_bucket_size():

    print(f"Getting {GLONET_BUCKET} bucket size...")

    client = Minio(
        endpoint=environ["AWS_S3_ENDPOINT"],
        access_key=environ["AWS_ACCESS_KEY_ID"],
        secret_key=environ["AWS_SECRET_ACCESS_KEY"],
        session_token=environ["AWS_SESSION_TOKEN"],
    )
    try:
        objects = client.list_objects(GLONET_BUCKET, recursive=True)
        bucket_size: int = sum(map(lambda obj: obj.size, objects))
        print(
            f"Total size of {GLONET_BUCKET} bucket: {_human_readable_size(bucket_size)} bytes"
        )
    except Exception as e:
        print(f"Cloud not compute {GLONET_BUCKET} bucket size: {e}")


def _clean_up_catalogs(
    forecast_informations_to_resolve: list[ForecastInformation],
):
    print("Cleaning catalogs...")
    glonet_daily_forecast_catalog_url = (
        f"{EDITO_DATA_CATALOG_URL}/glonet/glonet_1_4_daily_forecast"
    )
    response = get(glonet_daily_forecast_catalog_url)
    assert response.ok, response.text
    days_to_keep = list(
        map(
            lambda forecast_information: forecast_information.forecast_initial_day,
            forecast_informations_to_resolve,
        )
    )
    glonet_daily_forecast_catalog = response.json()
    links_of_catalogs_to_delete = filter(
        lambda catalog: catalog["rel"] == "child"
        and not any(map(lambda day: day in catalog["title"], days_to_keep)),
        glonet_daily_forecast_catalog["links"],
    )
    catalog_urls_to_delete = list(
        map(lambda links: links["href"], links_of_catalogs_to_delete)
    )
    print(f"Deleting {len(catalog_urls_to_delete)} catalogs...")
    list(map(_delete_edito, catalog_urls_to_delete))
    response = get(glonet_daily_forecast_catalog_url)
    assert response.ok, response.text
    forecast_catalog_count = len(response.json()["hasPart"])
    print(
        f"Catalogs cleaned, remaining {forecast_catalog_count} forecast catalogs"
    )


def _is_feature_to_delete(
    forecast_informations_to_resolve: list[ForecastInformation],
    feature_json: dict[str, Any],
):
    identifier: str = feature_json["properties"]["productIdentifier"]
    coverages_to_keep: list[str] = list(
        map(
            lambda forecast_information: f"{forecast_information.forecast_start_day}-{forecast_information.forecast_end_day}",
            forecast_informations_to_resolve,
        )
    )
    return identifier.startswith("glonet_1_4_daily_forecast-") and not any(
        map(
            lambda coverage: identifier.endswith(f"-{coverage}"),
            coverages_to_keep,
        )
    )


def _clean_up_features(
    forecast_informations_to_resolve: list[ForecastInformation],
):
    print("Cleaning features...")
    all_feature_url = f"{EDITO_DATA_API_URL}/search?owner=qgaudel"
    response = get(f"{all_feature_url}&limit=180")
    assert response.ok, response.text
    all_features = response.json()["features"]

    features_to_delete = filter(
        partial(_is_feature_to_delete, forecast_informations_to_resolve),
        all_features,
    )
    feature_urls_to_delete = list(
        map(
            lambda feature: next(
                link["href"].rpartition("/")[0]
                + "/"
                + feature["properties"]["productIdentifier"]
                for link in feature["links"]
                if link["rel"] == "self"
            ),
            features_to_delete,
        )
    )
    print(f"Deleting {len(feature_urls_to_delete)} features...")
    list(map(print, feature_urls_to_delete))
    response = get(f"{all_feature_url}&limit=180")
    assert response.ok, response.text
    feature_count = response.json()["numberMatched"]
    print(f"Features cleaned, remaining {feature_count} features")


def _clean_up_files(
    forecast_informations_to_resolve: list[ForecastInformation],
):
    print("Cleaning files...")
    public_forecast_folder_key = "public/glonet_1_4_daily_forecast"
    public_files: list[dict[str, str]] = list_objects(
        GLONET_BUCKET, public_forecast_folder_key
    )
    days_to_keep = list(
        map(
            lambda forecast_information: forecast_information.forecast_initial_day,
            forecast_informations_to_resolve,
        )
    )
    files_to_delete = filter(
        lambda file: not any(
            map(
                lambda day: file["Key"].startswith(
                    f"{public_forecast_folder_key}/{day}"
                ),
                days_to_keep,
            )
        ),
        public_files,
    )
    file_keys_to_delete = list(
        map(
            lambda file: file["Key"],
            files_to_delete,
        )
    )
    print(f"Deleting {len(file_keys_to_delete)} file...")
    list(map(partial(delete_object, GLONET_BUCKET), file_keys_to_delete))
    public_files: list[dict[str, str]] = list_objects(
        GLONET_BUCKET, public_forecast_folder_key
    )
    public_files_count = len(public_files)
    print(f"Files cleaned, remaining {public_files_count} public files")


def _clean_up_data(
    forecast_informations_to_resolve: list[ForecastInformation],
):
    _clean_up_catalogs(forecast_informations_to_resolve)
    _clean_up_features(forecast_informations_to_resolve)
    _clean_up_files(forecast_informations_to_resolve)


def main():

    _share_s3_public_directory()

    forecast_informations_to_resolve = _get_forecast_informations_to_resolve()

    _publish_glonet_root_catalog()
    _publish_glonet_1_4_daily_forecast_catalog()
    _publish_catalogs_and_features(forecast_informations_to_resolve)
    _clean_up_data(forecast_informations_to_resolve)

    _print_bucket_size()


main()
