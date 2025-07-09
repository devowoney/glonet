import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
import time
from torch.cuda.amp import autocast

def downloadCopernicusData()

def divid_input_nc(input_nc) -> xr.Dataset

def create_forecast(input_nc : str, 
                    model : str,
                    cycle : int ) -> xr.Dataset :
    input_data = xr.open_dataset(input_nc)
    input_date = input_data['time'].values[0]