#!/bin/bash
set -e

# setup ros environment
. "/opt/ros/$ROS_DISTRO/setup.bash"
/alc/webgme/bin/deepforge start --worker "$@"
