#!/bin/bash

if [ $# == 0 ]; then
    echo 'Building GPU docker'
    DOCKER_BUILDKIT=1 docker build --build-arg TAG=gpu-jupyter -t bluerov_sim:tf2.6.2.gpu .
fi
if [ $1 == 'cpu' ]; then
    echo 'Building CPU docker'
    DOCKER_BUILDKIT=1 docker build --build-arg TAG=jupyter -t bluerov_sim:tf2.6.2 .
else
    echo 'Invalid argument!'
    echo 'Use "build_docker.sh" for Tensorflow2 GPU or "build_docker.sh cpu" for Tensorflow2 CPU docker'
fi