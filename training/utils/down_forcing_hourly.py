import cdsapi
from pathlib import Path
import argparse

"""Download Copernicus Climate Data Store (CDS) products for specified years.
Download parameters retrieve reanalysis data for the entire globe at 12:00 UTC.
https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview
"""

DATASET = "reanalysis-era5-single-levels"


def product_request(product_types: list[str], 
                    years: list[str], ) -> dict:
    """Build CDS API request for daily reanalysis data (in NetCDF format) for specified years.

    Download variables:
        10m u components of wind, 10m v component of wind, 2m temperature,
        mean sea level pressure, surface pressure.

    Args:
        product_types (list[str]): product types to download
        years (list[str]): years to download

    Returns:
        dict: CDS request payload
    """
    
    return {
        "product_type": product_types,
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_pressure",
        ],
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
        "time": ["12:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
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
        nargs="+",
        default=["reanalysis"],
        help=(
            "Product types to download (choose among: reanalysis, "
            "ensemble_members, ensemble_mean, ensemble_spread). "
            "Example: -p reanalysis ensemble_mean or -p reanalysis,ensemble_mean."
        ),
    )
    parser.add_argument(
        "--years",
        "-y",
        nargs="+",
        required=True,
        help="Years to download (e.g., -y 1993 1994 or -y 1993,1994).",
    )
    parser.add_argument("--out_path", "-o",
                        dest = "out_path",
                        type = Path,
                        required = False,
                        help = "Output file path.")
    
    return parser.parse_args()

def main() :
    
    args = parse_args()

    years_input = args.years
    if len(years_input) == 1 and "," in years_input[0]:
        years = [y.strip() for y in years_input[0].split(",") if y.strip()]
    else:
        years = [str(y).strip() for y in years_input if str(y).strip()]

    product_input = args.product_type
    if len(product_input) == 1 and "," in product_input[0]:
        product_types = [p.strip() for p in product_input[0].split(",") if p.strip()]
    else:
        product_types = [str(p).strip() for p in product_input if str(p).strip()]

    # Download data
    request = product_request(product_types, years)
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