#!/bin/bash

docker run                            \
    --rm                              \
    -it                               \
    --name ros-master                 \
    --network ros                     \
    --ip 172.18.0.2                   \
    --hostname ros-master             \
    --add-host $(hostname):172.18.0.1 \
    ros:kinetic-ros-core roscore
