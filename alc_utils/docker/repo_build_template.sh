#!/bin/bash
if [[ -z "${REGISTRY_ADDR}" ]]; then
  if [[ -z "${ALC_REGISTRY_ADDR}" ]]; then
    echo "cannot find registry address. exitting."
    exit 1
  else
    export REGISTRY_ADDR="${ALC_REGISTRY_ADDR}"
  fi
fi

##Set the required environment variables
export ALC_HOME=$PWD
#
#

#Download the docker images (if any from the registry)
#export IMAGE_NAME=
#if [[ "$(docker images -q $IMAGE_NAME:latest 2> /dev/null)" == "" ]]; then
#  docker pull $REGISTRY_ADDR/$IMAGE_NAME:latest
#  docker tag $REGISTRY_ADDR/$IMAGE_NAME:latest $IMAGE_NAME:latest
#fi


##internal build scripts from the repo



#create any  docker network
#export docker_network_name= 
#echo "[ START ] Creating ROS Docker Network"
#docker network inspect $docker_network_name > /dev/null 2>&1
#if [ $? -eq 0 ]; then
#    echo "[SKIPPED] ROS Docker Network Already Exists"
#else
    #docker network create      \
    #    --gateway 172.18.0.1   \
    #    --subnet 172.18.0.0/16 \
    #    docker_network_name
    #echo "[SUCCESS] Created $docker_network_name Docker Network"
#fi

