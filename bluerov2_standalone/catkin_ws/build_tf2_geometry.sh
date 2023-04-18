#!/bin/bash
mkdir /tf2_ros/
pushd /tf2_ros/
mkdir src/
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool update src/geometry2
wstool up
source /opt/ros/melodic/setup.bash
rosdep install --from-paths src --ignore-src -y -r
catkin_make --cmake-args \
         -DCMAKE_BUILD_TYPE=RELEASE \
         -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.6 \
         -DPYTHON_INCLUDE_DIR:PATH=/usr/include/python3.6 \
         -DPYTHON_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
popd
#catkin_make --cmake-args \
#        -DCMAKE_BUILD_TYPE=RELEASE \
#        -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.7 \
#        -DPYTHON_INCLUDE_DIR:PATH=/usr/include/python3.7 \
#        -DPYTHON_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libpython3.7m.so

