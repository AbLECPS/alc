#!/bin/bash
source $ALC_HOME/bluerov2_standalone/catkin_ws/devel/setup.bash
source /opt/ros/melodic/setup.bash
rosrun rviz rviz -d $ALC_HOME/bluerov2_standalone/catkin_ws/src/vandy_bluerov/rvizbluerov2_control_rtreach.rviz
