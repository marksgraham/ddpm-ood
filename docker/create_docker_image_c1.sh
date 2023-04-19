#!/bin/bash
#
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
TAG=nvcr.io/r5nte7msx1tj/amigo/ddp-ood:v0.1.2

cp ../requirements.txt .
docker build --network=host --tag "${TAG}" . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

#push
docker push "${TAG}"
