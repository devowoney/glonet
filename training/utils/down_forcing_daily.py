import cdsapi
from pathlib import Path
import argparse

"""Download Copernicus Climate Data Store (CDS) products for specified years.
Download parameters retrieve reanalysis daily mean data for the entire globe at 00:00 UTC.
https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=download
"""

DATASET = "derived-era5-single-levels-daily-statistics"
VALID_VARIABLES = [
    "10m_u_component_of_wind", 
    "10m_v_component_of_wind", 
    "2m_temperature", 
    "mean_sea_level_pressure", 
    "surface_pressure", 
]


def product_request(product_types : str, 
                    years : str, 
                    variable : str) -> dict :
    """Build CDS API request for daily reanalysis data (in NetCDF format) for specified years.

    Download variables:
        10m u components of wind, 10m v component of wind, 2m temperature,
        mean sea level pressure, surface pressure.

    Args:
        product_types (str): product types to download
        years (str): years to download

    Returns:
        dict: CDS request payload
    """
    
    return {
        "product_type": product_types,
        "variable": variable,
        "year": years,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31",
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        "area": [90, -180, -80, 180],
    }


def parse_args () :
    parser = argparse.ArgumentParser(
        description="""Download Copernicus Climate data 
        (ERA5 post-processed daily statistics on single levels from 1940 to present) for GLONET training.
        Download variables:
        10m u components of wind, 10m v component of wind, 2m temperature,
        mean sea level pressure, surface pressure.
        """
    )
    parser.add_argument(
        "--product_type",
        "-p",
        default=["reanalysis"],
        help="Product types to download (choose among: reanalysis, ensemble_members, ensemble_mean). "
    )
    parser.add_argument(
        "--year",
        "-y",
        required = True,
        help="Years to download (only one year accepted).",
    )
    parser.add_argument(
        "--variable", "-v",
        type = str,
        required = True,
        choices = VALID_VARIABLES,
        help = "Variable to download."
    )
    parser.add_argument(
        "--out_path", "-o",
        type = Path,
        help = "Output file path.")
    
    return parser.parse_args()

def main() :
    
    args = parse_args()
    

    # Download data
    request = product_request(args.product_type, args.year, args.variable)
    client = cdsapi.Client()
    retrieval = client.retrieve(DATASET, request)
    
    if args.out_path:
        out_path = args.out_path.expanduser().resolve() # Absolute path
        retrieval.download(str(out_path))
        print(f"Copernicus Marine data is completely downloaded and concatenated in < {out_path} >")
    else:
        retrieval.download()


if __name__ == "__main__":
    main()