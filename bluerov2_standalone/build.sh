#!/bin/bash

BASEDIR=$(dirname "$0")
pushd $BASEDIR/catkin_ws/
source build_sources.sh
popd
