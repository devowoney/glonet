#!/bin/bash

if [ ! -e "/home/onyxia/miniforge3/etc/profile.d/conda.sh" ]; then

    # Install minconda
    echo | curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh


    exec bash

else
    source ~/miniforge3/etc/profile.d/conda.sh
    
    # Install pakages for glonet from env_yaml file 
    conda update -n base -c conda-forge conda
    conda env create -n glon --file ./glonet_daily_forecast_data_orchestration/conda_environment.yml
    conda init

    # Creat jupyter_notebook environment from conda environment
    # If you use jupyter service, activate (uncomment) the command line below :
    # ipython kernel install --user --name=glon

    # ADD Cuda variables
    echo "export CUDA_HOME=/usr/local/cuda" >> ../.bashrc
    echo "export PATH=$CUDA_HOME/bin:$PATH" >> ../.bashrc
    echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH" >> ../.bashrc

    # ADD user alias
    echo "alias c0='clear'" >> ../.bashrc
    echo "alias ..='cd ../'" >> ../.bashrc
    echo "alias ...='cd ../../'" >> ../.bashrc
    echo "alias lr='ls -alrth'" >> ../.bashrc
    echo "alias lR='ls -alRth'" >> ../.bashrc
    echo "alias r='more'" >> ../.bashrc

    # Restart bash
    source ../.bashrc
    
fi