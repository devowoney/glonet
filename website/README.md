# GLONET: Mercator Ocean's AI system

GLONET is a neural-network ocean system developed by [Mercator Ocean international](https://www.mercator-ocean.eu/) as part of the [EDITO-ModelLab project](https://edito-modellab.eu/).

##  Get started

[table of content](./_sidebar.md ':include')

## GLONET

GLONET is an awesome Neural Network.

### Publications

- [A. El Aouni et al, 2024, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM](https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1524960).

## GLONET on EDITO

Some GLONET capabilities as been ported to the [EDITO platform](https://dive.edito.eu/):

0. [Daily forecasts](#daily-forecasts) (available publicly)
0. [On-demand forecasts](#on-demand-forecasts) (restricted to research partners)

### Daily forecasts

You can browse the forecasts assets at **1/4 resolution** on [EDITO viewer](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/glonet/glonet_1_4_daily_forecast).
The **10-day forecasts** are **updated daily** on the platform, using the [daily mean fields from Global Ocean Physics Analysis and Forecast](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/copernicus-marine-products/copernicus-marine-product-GLOBAL_ANALYSISFORECAST_PHY_001_024/copernicus-marine-dataset-cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m) of the day as initial state of the forecast.
The forecast of the **last 10 days are kept in the EDITO data lake**, but forecast of the **anterior days can be reproduced on-demand**.

Among other information, the metadata exposes links allowing you to continue the exploration of the data inside a Jupterlab instance running on the [EDITO datalab](https://datalab.dive.edito.eu/).

### On-demand forecasts

An EDITO process is available to (re)produced GLONET forecast.
At launch time, you can choose:

0. The number of days to forecast
0. Which data to use as initial state of the forecast
0. Where to store the output data

To propriety reason, the access to this process is **restricted to identified research partners**.
If you wish to run GLONET on-demand on EDITO, please [contact us](contact.md).

### Architecture blueprint

This section focuses on explaining how GLONET capabilities have been ported to EDITO.

## Contact

For any information about GLONET, you can contact...
