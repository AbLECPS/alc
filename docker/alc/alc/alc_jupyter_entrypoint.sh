#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
echo $JUPYTER_WORK_DIR
cd $JUPYTER_WORK_DIR
echo "current directory" 
pwd
echo "running jupyter command"
jupyter notebook --allow-root --config=/jupyter_notebook_config.py
