from get_inits import generate_initial_data
from glonet_forecast import generate_forecast_file
from model import synchronize_model_locally

EDITO_BUCKET_NAME="project-da-ml"
URL="https://minio.dive.edito.eu/project-glonet/public/glonet_1_4_daily_forecast/2025-07-01/GLONET_MOI_2025-07-02_2025-07-11.nc"

def main() :
    # Divid inital states array in 3 files
    cif1, cif2, cif3 = generate_initial_data(bucket_name=EDITO_BUCKET_NAME, 
                                             forecast_netcdf_file_url=URL)
    
    # Sync weight of the model <- Testing already existing file
    local_dir = "../weight"
    synchronize_model_locally(local_dir=local_dir)
    
    generate_forecast_file(bucket_name=EDITO_BUCKET_NAME, 
                           forecast_netcdf_file_url=URL, 
                           initial_file_1_url=cif1, 
                           initial_file_2_url=cif2, 
                           initial_file_3_url=cif3)
    
if __name__ == "__main__" :
    main()

    