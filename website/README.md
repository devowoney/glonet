# GLONET: Mercator's End-to-End Neural Forecasting System

GLONET is a neural-network ocean system developed by [Mercator Ocean international](https://www.mercator-ocean.eu/) as part of the [EDITO-ModelLab project](https://edito-modellab.eu/).

<!-- ## Get started -->

<!-- [table of content](./_sidebar.md ':include') -->

## Introduction to GLONET

Accurate ocean forecasting is crucial in different areas ranging from science to decision making. 
Recent advancements in data-driven models have shown significant promise, particularly in weather forecasting community, but yet no data-driven approaches have matched the accuracy and the scalability of traditional global ocean forecasting systems that rely on physics-driven numerical models and can be very computationally expensive, depending on their spatial resolution or complexity.
Here, we introduce GLONET, a global ocean neural network-based forecasting system, developed by Mercator Ocean International.
GLONET is trained on the global Mercator Ocean physical reanalysis GLORYS12 to integrate physics-based principles through neural operators and networks, which dynamically capture local-global interactions within a unified, scalable framework, ensuring high small-scale accuracy and efficient dynamics.
GLONET's performance is assessed and benchmarked against two other forecasting systems: the global Mercator Ocean analysis and forecasting 1/12 high-resolution physical system GLO12 and a recent neural-based system also trained from GLORYS12.
A series of comprehensive validation metrics is proposed, specifically tailored for neural network-based ocean forecasting systems, which extend beyond traditional point-wise error assessments that can introduce bias towards neural networks optimized primarily to minimize such metrics.
The preliminary evaluation of GLONET shows promising results, for temperature, sea surface height, salinity and ocean currents.
GLONET's experimental daily forecast are accessible through the European Digital Twin Ocean platform EDITO.

### Architecture overview

<div align="center">
<figure>
<img
src="https://minio.dive.edito.eu/project-glonet/public/glonet_thumbnail.png"
alt="GLONET architecture overview" width="800">
<figcaption>Overview of GLONET’s architecture containing different modules, particularly time-block designed
to learn feature maps encapsulating initial conditions along with forcings. A spatial module architectured to
learn multi-scale dynamics, and finally and encoder-decoder to fuse multi-scale circulations into a unified
latent space.</figcaption>
</figure>
</div>

### Output variables

#### Surface height

The 2D sea surface height above geoid, as defined in the [Climate Forecast Convention](https://cfconventions.org/).

![Sea surface height above geoid](/assets/zos_ortho.png ':size=30%')

#### Temperature

The 3D sea water potential temperature, as defined in the [Climate Forecast Convention](https://cfconventions.org/).

![Sea water potential temperature](/assets/thetao_ortho.png ':size=30%')

#### Salinity

The 3D sea water salinity, as defined in the [Climate Forecast Convention](https://cfconventions.org/).

![Sea water salinity](/assets/so_ortho.png ':size=30%')

#### Current

The 3D eastward sea water velocity and northward sea water velocity, as defined in the [Climate Forecast Convention](https://cfconventions.org/).

![Eastward sea water velocity](/assets/uo_ortho.png ':size=30%')
![Northward sea water velocity](/assets/vo_ortho.png ':size=30%')

### Publications

- [A. El Aouni et al, 2024, AGU24, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM](https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1524960).
- [A. El Aouni et al, 2024, Preprint, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM](https://arxiv.org/abs/2412.05454).

## GLONET on EDITO

Some GLONET capabilities as been ported to the [EDITO platform](https://dive.edito.eu/):

0. [Daily forecasts](#daily-forecasts) (available publicly)
0. [On-demand forecasts](#on-demand-forecasts) (restricted to research partners)

### Daily forecasts

You can browse the forecasts assets at **1/4 resolution** on [EDITO viewer](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/glonet/glonet_1_4_daily_forecast).

[GLONET daily forecasts on EDITO viewer](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/glonet/glonet_1_4_daily_forecast ':include :type=iframe width=100% height=600px')

The **10-day forecasts** are **updated daily** on the platform, using the [daily mean fields from Global Ocean Physics Analysis and Forecast](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/copernicus-marine-products/copernicus-marine-product-GLOBAL_ANALYSISFORECAST_PHY_001_024/copernicus-marine-dataset-cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m) of the day as initial state of the forecast.
The forecast of the **last 15 days are kept in the EDITO data lake**, but forecasts of the **15 anterior days can be reproduced on-demand**.

Among other information, the metadata exposes links allowing you to continue the exploration of the data inside a Jupterlab instance running on the [EDITO datalab](https://datalab.dive.edito.eu/).

#### Status

You can check the daily forecast status [here](https://glonet.lab.dive.edito.eu/status).

### On-demand forecasts

An EDITO process is available to (re)produced GLONET forecast.
At launch time, you can choose:

0. The number of days to forecast
0. Which data to use as initial state of the forecast
0. Where to store the output data

For propriety reason, the access to this process is **restricted to identified research partners**.
If you wish to run GLONET on-demand on EDITO, please [contact us](#contact).

## Contact

For any information about GLONET, you can contact [glonet@mercator-ocean.eu](mailto:glonet@mercator-ocean.eu).
