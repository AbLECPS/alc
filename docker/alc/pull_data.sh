#!/bin/bash
docker run --rm  -e ALC_WORKING_DIR=$ALC_WORKING_DIR -v $ALC_WORKING_DIR:$ALC_WORKING_DIR alc_data:latest bash -c "source file.sh"
pushd $ALC_HOME/bluerov2_standalone/catkin_ws
./pull_bluerov2_description
popd
