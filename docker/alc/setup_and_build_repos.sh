#!/bin/bash
set -e
set -o pipefail

if [ $# -eq 0 ];
  then
    echo "No arguments supplied. Input argument should be one of the following ALL / alc_repo / bluerov2_standalone"
    exit
fi;

ALC_HOME=$ALC_HOME
ALC_WORKING_DIR=$ALC_WORKING_DIR
ALC_DOCKERFILES=$ALC_DOCKERFILES
ALC_DOCKER_NETWORK_GATEWAY="172.23.0.1"
ALC_DOCKER_NETWORK="alcnet"
ALC_GITSERVER_HOST=$ALC_DOCKER_NETWORK_GATEWAY
ALC_GITSERVER_ROOT=$ALC_WORKING_DIR/.gitserver
ALC_GITSERVER_PORT=2222
git_server_started=0

if [ ! "$(docker ps -q -f name=alc_gitserver)" ]; 
then  
  echo "Starting git server to configure ssh"
  docker run -d --rm --name alc_gitserver \
  -p $ALC_GITSERVER_PORT:22 \
  -v $ALC_GITSERVER_ROOT/keys:/git-server/keys \
  -v $ALC_GITSERVER_ROOT/repos:/git-server/repos \
  --network $ALC_DOCKER_NETWORK \
  jkarlos/git-server-docker
  
  git_server_started=1

fi

if [ "$(docker ps -q -f name=alc_alc)" ];
then 
    echo "Using existing alc docker!"
    docker exec -it alc_alc \
        bash -c "$ALC_HOME/docker/alc/setup_and_build_repos_ep.sh \"$@\"" 
else
    echo "Starting new instance of alc docker!"
    docker run -it --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v $ALC_HOME:$ALC_HOME \
        -v $ALC_WORKING_DIR:$ALC_WORKING_DIR \
        -v $ALC_DOCKERFILES:$ALC_DOCKERFILES \
        -e ALC_HOME=$ALC_HOME \
        -e ALC_DOCKERFILES=$ALC_DOCKERFILES \
        -e ALC_WORKING_DIR=$ALC_WORKING_DIR \
        -e ALC_GITSERVER_HOST=$ALC_GITSERVER_HOST \
        -e ALC_GITSERVER_PORT=$ALC_GITSERVER_PORT \
        alc:latest \
        bash -c "$ALC_HOME/docker/alc/setup_and_build_repos_ep.sh \"$@\"" 
fi

if [ $git_server_started -eq 1 ];then
  
  echo "Stopping gitserver"
  docker stop alc_gitserver
  
fi;

