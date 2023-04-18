#!/bin/sh

#add symlink for sensor meshes
sudo ln -sf $PWD/src/bluerov2/bluerov2_description/meshes/pressure.dae /opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/pressure.dae
sudo ln -sf $PWD/src/bluerov2/bluerov2_description/meshes/dvl.dae /opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/dvl.dae

#add symlink for workspace to root
sudo rm /aa
sudo ln -sf $PWD /aa

sudo ln -sf /usr/bin/python /usr/local/bin/python
. ${PWD}/devel/setup.bash
. /opt/ros/melodic/setup.bash --extend
. /usr/share/gazebo-9/setup.sh --extend
. /usr/share/gazebo/setup.sh --extend
export ROS_MASTER_URI='http://ros-master:11311'
export GAZEBO_MASTER_URI='http://aa_uuvsim:11345'
export PYTHONPATH='/aa/devel/lib/python2.7/dist-packages':$PYTHONPATH

. ${PWD}/src/vandy_bluerov/scripts/run_rviz_bluerov2.sh

