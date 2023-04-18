#!/bin/bash

mkdir results

DOCKER_BUILDKIT=1 docker build -t bluerov_sim:tf2.6.2.gpu .

docker run                             \
    --rm                               \
    -v $PWD:/aa                        \
    alc:latest sh -c "                \
        cd /aa/src;                    \
        catkin_init_workspace;         \
        cd /aa;                        \
        catkin_make;                   \ 
    "
