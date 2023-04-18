#!/bin/bash
mkdir -p /cvbridge_build_ws/src
pushd /cvbridge_build_ws
catkin init
pushd src
git clone -b noetic https://github.com/ros-perception/vision_opencv.git
sed -i 's/find_package(Boost REQUIRED python37)/find_package(Boost REQUIRED python3)/g' /cvbridge_build_ws/src/vision_opencv/cv_bridge/CMakeLists.txt

cat /cvbridge_build_ws/src/vision_opencv/cv_bridge/CMakeLists.txt
popd
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3.6 \
              -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
              -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
source /opt/ros/melodic/setup.bash
catkin build cv_bridge
popd

