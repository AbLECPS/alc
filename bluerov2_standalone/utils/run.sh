#!/bin/bash
pushd $ALC_HOME/catkin_ws
. setup_gazebo_ros_connection.sh
. $ALC_WORKING_DIR/execution/gazebo/setup_env.sh
popd
echo "starting xpra"
. /opt/ros/kinetic/setup.bash --extend
export ROS_MASTER_URI=http://ros-master:11311
xterm &
rviz -d /aa/src/vandy_bluerov/rviz/bluerov2_control.rviz &
echo "finished starting xpra"

