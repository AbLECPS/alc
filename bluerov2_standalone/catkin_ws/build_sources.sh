#!/bin/bash

mkdir results

docker run                             \
    --rm                               \
    -v $PWD:/aa                        \
    alc:latest /bin/sh -c "       \
        . /opt/ros/melodic/setup.bash  \
        cd /aa/src;                    \
        catkin_init_workspace;         \
        cd /aa;                        \
        catkin_make;                   \ 
    "
mkdir $PWD/src/vandy_bluerov/results

. ${PWD}/pull_sensor_meshes.sh
. ${PWD}/pull_bluerov2_description.sh