#!/bin/bash

# Exit on error, don't suppress
set -e

# Execute job
pushd $ALC_WORKING_DIR/{{ relative_result_dir|string }}
source /opt/ros/$ROS_DISTRO/setup.bash
python2.7 main.py
popd
