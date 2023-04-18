#!/bin/bash

if [[ -z "${REGISTRY_ADDR}" ]]; then
  if [[ -z "${ALC_REGISTRY_ADDR}" ]]; then
    echo "cannot find registry address. exitting."
    exit 1
  else
    export REGISTRY_ADDR="${ALC_REGISTRY_ADDR}"
  fi
fi


if [[ "$(docker images -q alc:latest 2> /dev/null)" == "" ]]; then
  docker pull $REGISTRY_ADDR/alc:latest
  docker tag $REGISTRY_ADDR/alc:latest alc:latest
  docker tag $REGISTRY_ADDR/alc:latest bluerov_sim:tf2.6.2.gpu
fi


pushd $REPO_HOME/catkin_ws
echo "[ START ] Building BlueROV2_Standlone Sources"
./build_sources.sh
echo "[ DONE ] Building BlueROV2_Standlone Sources"
popd


echo "[ START ] Creating ROS Docker Network"
docker network inspect ros > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[SKIPPED] ROS Docker Network Already Exists"
else
    docker network create      \
        --gateway 172.18.0.1   \
        --subnet 172.18.0.0/16 \
        ros
    echo "[SUCCESS] Created ROS Docker Network"
fi

