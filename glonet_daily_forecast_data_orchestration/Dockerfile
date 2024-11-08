FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER .condarc /tmp/.condarc
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_environment.yml /tmp/conda_environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pip.conf /tmp/pip.conf
RUN /usr/local/bin/_entrypoint.sh micromamba install -y -n base -f /tmp/conda_environment.yml

ADD --chown=$MAMBA_USER:$MAMBA_USER src /src

CMD [ "python", "/src/glonet_edito_data_orchestration.py" ]
