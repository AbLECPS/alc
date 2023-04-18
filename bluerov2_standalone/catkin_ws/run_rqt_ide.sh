#!/bin/bash

#only do this once...
if [ ! -f ONE_TIME_SETUP ]; then

  #update gazebo-9
  sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
  wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys AB17C654
  apt update && sudo apt install -y libgazebo9-dev gazebo9 ros-melodic-gazebo9-*
  apt install ros-melodic-rqt-py-trees -y
  apt-get install -y ros-melodic-rviz-visual-tools ros-melodic-moveit-visual-tools ros-melodic-moveit-ros-visualization

touch ONE_TIME_SETUP

#add symlink for workspace to root
ln -sf $ALC_HOME/catkin_ws /aa


ln -sf /usr/bin/python /usr/local/bin/python

fi

. ${PWD}/devel/setup.bash
. /opt/ros/melodic/setup.bash --extend
. /usr/share/gazebo-9/setup.sh --extend
. /usr/share/gazebo/setup.sh --extend
export ROS_MASTER_URI='http://ros-master:11311'
export GAZEBO_MASTER_URI='http://aa_uuvsim:11345'
export PYTHONPATH='/aa/devel/lib/python2.7/dist-packages':$PYTHONPATH

rqt_py_trees