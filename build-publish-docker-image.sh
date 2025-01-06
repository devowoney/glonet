#!/usr/bin/env bash

set -eo pipefail

export REGISTRY=docker.mercator-ocean.fr

REGISTRY_URI="https://${REGISTRY}"
REGISTRY_REPOSITORY=${REGISTRY}/moi-docker/glonet

docker login ${REGISTRY_URI}

CONTAINER_IMAGE_NAME=glonet_daily_forecast_orchestration
CONTAINER_IMAGE_TAG=0.0.17

if docker manifest inspect ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG} > /dev/null ; then
    echo "Image tag found on the Nexus, skipping build and publish"
else
    echo "Image tag does not exists on Nexus"
    docker build --ulimit nofile=65536:65536 --tag ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG} --platform linux/amd64 glonet_daily_forecast_data_orchestration
    docker push ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG}
fi
echo "Image link: $REGISTRY_REPOSITORY/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG"

CONTAINER_IMAGE_NAME=glonet_daily_forecast_orchestration_cronjob
CONTAINER_IMAGE_TAG=0.0.13

if docker manifest inspect ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG} > /dev/null ; then
    echo "Image tag found on the Nexus, skipping build and publish"
else
    echo "Image tag does not exists on Nexus"
    docker build --ulimit nofile=65536:65536 --tag ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG} --platform linux/amd64 glonet_daily_forecast_data_orchestration_cronjob
    docker push ${REGISTRY_REPOSITORY}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG}
fi
echo "Image link: $REGISTRY_REPOSITORY/$CONTAINER_IMAGE_NAME:$CONTAINER_IMAGE_TAG"
