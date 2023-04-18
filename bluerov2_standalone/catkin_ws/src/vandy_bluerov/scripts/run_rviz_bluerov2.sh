#!/bin/bash
echo $PWD
source devel/setup.bash
source /opt/ros/melodic/setup.bash
rosrun rviz rviz -d src/vandy_bluerov/rviz/bluerov2_control.rviz
