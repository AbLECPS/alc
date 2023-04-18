#!/bin/bash

echo "172.18.0.2    ros-master" >> /etc/hosts
echo "172.18.0.4    aa_uuvsim" >> /etc/hosts

echo "export ALC_HOME=$REPO_HOME/bluerov2_standalone" >> ~/.bashrc

source ~/.bashrc

pushd $ALC_HOME/catkin_ws
source build_sources.sh
popd

docker network connect --ip 172.18.0.7  ros  alc_codeserver

echo "IDE setup complete"