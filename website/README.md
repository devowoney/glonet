<div align="center">

# GLONET

**Mercator's End-to-End Neural Forecasting System**

<img width="200" src="https://minio.dive.edito.eu/project-glonet/public/glonet_thumbnail.png" alt="logo of GLONET">

</div>

GLONET is:

- A neural network-based ocean forecasting system developed by [Mercator Ocean international](https://www.mercator-ocean.eu/), which architecture is described in the [scientific paper](#publications), outputting in seconds a 10-days daily forecasts of the temperature, sea surface height, salinity and currents at 1/4 degree resolution ([1/12 degree effective resolution](#effective-resolution)) over 21 depth levels.
- A port of the forecasting system on the European Digital Twin Ocean platform [EDITO](https://edito.eu) producing 10-days forecasts on a daily basis.
- 10-days forecast data products discoverable and exploitable on the European Digital Twin Ocean.

## Content

[table of content](./_sidebar.md ":include")

## Context and conception

### Abstract

Accurate ocean forecasting is crucial in different areas ranging from science to decision making.
Recent advancements in data-driven models have shown significant promise, particularly in weather forecasting community, but yet no data-driven approaches have matched the accuracy and the scalability of traditional global ocean forecasting systems that rely on physics-driven numerical models and can be very computationally expensive, depending on their spatial resolution or complexity.

Here, we introduce GLONET, a global ocean neural network-based forecasting system, developed by Mercator Ocean International.
GLONET is trained on the global Mercator Ocean physical reanalysis [GLORYS12](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) to integrate physics-based principles through neural operators and networks, which dynamically capture local-global interactions within a unified, scalable framework, ensuring high small-scale accuracy and efficient dynamics.
GLONET's performance is assessed and benchmarked against two other forecasting systems: the global Mercator Ocean analysis and forecasting 1/12 high-resolution physical system [GLO12](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description) and a recent neural-based system also trained from [GLORYS12](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description).

A series of comprehensive validation metrics is proposed, specifically tailored for neural network-based ocean forecasting systems, which extend beyond traditional point-wise error assessments that can introduce bias towards neural networks optimized primarily to minimize such metrics.
The preliminary evaluation of GLONET shows promising results, for temperature, sea surface height, salinity and ocean currents.

GLONET's experimental daily forecast are accessible through the European Digital Twin Ocean platform EDITO.

### Architecture

<div align="center">
<figure>
<img
src="https://minio.dive.edito.eu/project-glonet/public/glonet_architecture.png"
alt="GLONET architecture overview" width="800">
<figcaption>Overview of GLONET’s architecture containing different modules, particularly time-block designed
to learn feature maps encapsulating initial conditions along with forcings. A spatial module architectured to
learn multi-scale dynamics, and finally and encoder-decoder to fuse multi-scale circulations into a unified
latent space.</figcaption>
</figure>
</div>

### Effective resolution

GLONET outputs data at 1/4 degree resolution, but it was trained on [GLORYS12](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description), a dataset at 1/12 degree resolution.
Hence, GLONET output represents a 1/12 degree resolution dynamics, which we call its **effective resolution**.

It is possible to obtain good results, depending on the use cases, by interpolating GLONET output to 1/12 degree resolution.
This interpolation can be done on-the-fly while only keeping GLONET 1/4 degree output in your storage, which is significantly lighter than 1/12 degree resolution datasets.

The following figures illustrate this but more details can be found in the [scientific paper](#publications).

<div align="center">
<figure>
<img-comparison-slider>
  <img slot="first" src="https://minio.dive.edito.eu/project-glonet/public/GLO12_square.png" />
  <img slot="second" src="https://minio.dive.edito.eu/project-glonet/public/GLONET_1_12_vs_GLO12_square.png" />
</img-comparison-slider>
<figcaption>Illustration of GLONET 1/12 degree effective resolution on sea surface current compared to <a href="https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description">GLO12</a> (1/12 degree resolution). Outside the white square is plotted a sample of GLO12. Inside the white square is plotted GLO12 on the left, and GLONET interpolated at 1/12 degree resolution on the right.</figcaption>
</figure>
</div>

<div align="center">
<figure>
<img-comparison-slider>
  <img slot="first" src="https://minio.dive.edito.eu/project-glonet/public/GLONET_1_12_vs_1_4_square.png" />
  <img slot="second" src="https://minio.dive.edito.eu/project-glonet/public/GLONET_1_12_vs_GLO12_square.png" />
</img-comparison-slider>
<figcaption>Illustration of GLONET 1/12 degree effective resolution on sea surface current compared to GLONET 1/4 degree native resolution. Inside the white square is plotted a sample of GLONET interpolated at 1/12 degree resolution. Outside the white square is plotted GLONET at 1/4 degree resolution on the left, and <a href="https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description">GLO12</a> (1/12 degree resolution) on the right.</figcaption>
</figure>
</div>

### Publications

- [A. El Aouni et al, 2024, AGU24, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM](https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1524960).
- [A. El Aouni et al, 2024, Preprint, GLONET: MERCATOR’S END-TO-END NEURAL FORECASTING SYSTEM](https://arxiv.org/abs/2412.05454).

## Forecasts

Some GLONET capabilities as been ported to the [EDITO platform](https://edito.eu/):

0. [Visualize the latest forecast](#latest-forecast) (available publicly)
1. [Browse and download all daily forecasts](#all-daily-forecasts) (available publicly)
2. [Execute on-demand forecasts](#on-demand-forecasts) (restricted to research partners)

The **10-day forecasts** are **updated daily** on the platform, using the [daily mean fields from Global Ocean Physics Analysis and Forecast](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/copernicus-marine-products/copernicus-marine-product-GLOBAL_ANALYSISFORECAST_PHY_001_024/copernicus-marine-dataset-cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m) of the day as initial state of the forecast.
The forecast of the **last 15 days are kept in the EDITO data lake**, but forecasts of the **15 anterior days can be reproduced on-demand**.

### Latest forecast

You can explore the latest forecast at **1/4 resolution** in [EDITO data explorer](https://my-ocean.dive.edito.eu/-/fia59e3gfn?tRelative=P10D&tStartOf=day).

#### Currents

<iframe src="https://my-ocean.dive.edito.eu/-/kjgy50v742?tRelative=P10D&tStartOf=day" width="100%" height="600px"></iframe>

#### Temperature

<iframe src="https://my-ocean.dive.edito.eu/-/qtu46z9p9b?tRelative=P10D&tStartOf=day" width="100%" height="600px"></iframe>

#### Sea surface height

<iframe src="https://my-ocean.dive.edito.eu/-/iqyb9xlbnb?tRelative=P10D&tStartOf=day" width="100%" height="600px"></iframe>

#### Salinity

<iframe src="https://my-ocean.dive.edito.eu/-/pfrsxptb9d?tRelative=P10D&tStartOf=day" width="100%" height="600px"></iframe>

### All daily forecasts

You can browse and fetch the forecasts assets at **1/4 resolution** [here](https://viewer.dive.edito.eu/map?catalog=https://api.dive.edito.eu/data/catalogs/glonet/glonet_1_4_daily_forecast).

Among other information, the metadata exposes links allowing you to continue the exploration of the data inside a Jupterlab instance running on the [EDITO datalab](https://datalab.dive.edito.eu/).

### On-demand forecasts

An EDITO process is available to (re)produced GLONET forecast.
At launch time, you can choose:

0. The number of days to forecast
1. Which data to use as initial state of the forecast
2. Where to store the output data

For propriety reason, the access to this process is **restricted to identified research partners**.
If you wish to run GLONET on-demand on EDITO, please [contact us](#contact).

### Production status

<iframe width="100%" height="70px" src="status/index.html" scrolling="no"/></iframe>

## Contact

For any information about GLONET, you can contact [glonet@mercator-ocean.eu](mailto:glonet@mercator-ocean.eu).
