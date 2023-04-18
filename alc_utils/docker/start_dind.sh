#!/usr/bin/env bash
export did=$1
export hname=codeserver$did
export ALC_DOCKERROOT=$ALC_WORKING_DIR/.docker
export dind_docker_folder=$ALC_DOCKERROOT/docker${did}
export dind_dockerlib_folder=$ALC_DOCKERROOT/docker${did}lib
export REPO_ROOT=$ALC_DOCKERROOT/.users/user${did}
export REPO_HOME=$REPO_ROOT
export REPO_WORKING_DIR=$ALC_WORKING_DIR
export REPO_DOCKERFILES=$ALC_DOCKERFILES

sudo mkdir -p $ALC_DOCKERROOT
sudo mkdir -p $dind_docker_folder
sudo mkdir -p $dind_dockerlib_folder
sudo mkdir -p $REPO_ROOT

if [[ -z "$ALC_DOCKER_NETWORK_GATEWAY" ]]; then
    echo "ALC_DOCKER_NETWORK_GATEWAY is not defined. Using default."
    export ALC_DOCKER_NETWORK_GATEWAY="172.23.0.1"
fi

if [[ -z "$ALC_REGISTRY_HOST" ]]; then
    echo "ALC_REGISTRY_HOST is not defined. Using default."
    export ALC_REGISTRY_HOST=$ALC_DOCKER_NETWORK_GATEWAY
fi

if [[ -z "$ALC_REGISTRY_PORT" ]]; then
    echo "ALC_REGISTRY_PORT is not defined. Using default."
    export ALC_REGISTRY_PORT=5001
fi

if [[ -z "$ALC_REGISTRY_ADDR" ]]; then
    echo "ALC_REGISTRY_ADDR is not defined. Using default."
    export ALC_REGISTRY_ADDR="$ALC_REGISTRY_HOST:$ALC_REGISTRY_PORT"
fi


docker run --runtime nvidia \
            -e REPO_HOME=$REPO_HOME \
            -e REPO_WORKING_DIR=$REPO_WORKING_DIR \
            -e REPO_DOCKERFILES=$REPO_DOCKERFILES \
            -e VSCODE_START_DIR=$REPO_HOME \
            -e username=alc \
            -e REGISTRY_ADDR=$ALC_REGISTRY_ADDR \
            -e ALC_DOCKERFILES=$ALC_DOCKERFILES \
            -e ALC_USERNAME=$did \
            -e GIT_SERVER_IP=$ALC_GITSERVER_HOST \
            -e GIT_SERVER_PORT=$ALC_GITSERVER_PORT \
            -e GIT_SERVER_URL=$ALC_GITSERVER_URL \
            -e ALC_SSH_PORT=$ALC_SSH_PORT \
            -e ALC_SSH_HOST=$ALC_SSH_HOST \
            -e ALC_HOME=$ALC_HOME \
            --name $hname \
            -v $dind_docker_folder:/var/run \
            -v $dind_dockerlib_folder:/var/lib/docker \
            -v $REPO_HOME:$REPO_HOME \
            -v $REPO_WORKING_DIR:$REPO_WORKING_DIR \
            -v $REPO_DOCKERFILES:$REPO_DOCKERFILES \
            -v $ALC_HOME:$ALC_HOME \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            --privileged --gpus all \
            --network alcnet \
            --hostname $hname \
            alc_dind