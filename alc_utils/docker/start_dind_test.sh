#!/usr/bin/env bash
export did=$1
export hname=codeserver$did
export ALC_DOCKERROOT=$ALC_WORKING_DIR/docker
export dind_docker_folder=$ALC_DOCKERROOT/docker${did}
export dind_dockerlib_folder=$ALC_DOCKERROOT/docker${did}lib
export REPO_ROOT=$ALC_DOCKERROOT/users/user${did}
export REPO_HOME=$REPO_ROOT
export REPO_WORKING_DIR=$ALC_WORKING_DIR
export REPO_DOCKERFILES=$ALC_DOCKERFILES

sudo mkdir -p $ALC_DOCKERROOT
sudo mkdir -p $dind_docker_folder
sudo mkdir -p $dind_dockerlib_folder
sudo mkdir -p $REPO_ROOT


#export REPO_ROOT=$ALC_HOME
#export ALC_DOCKERROOT=/hdd2/docker

#This may not be required because of registry
#replace with registry
#export dind_dockerimage_folder=/hdd2/docker/dockervs/image
#export dind_dockeroverlay_folder=/hdd2/docker/dockervs/overlay2
#export image_folder=/hdd2/images/
##


#Question should the following mapping be used 
#            -v $REPO_HOME:$REPO_HOME \
#            -v $REPO_WORKING_DIR:$REPO_WORKING_DIR \
#            -v $REPO_DOCKERFILES:$REPO_DOCKERFILES \
#Answer probably it should be in the dind
#However while starting code-server, this can be changed?

#also attach the hostname for alc_gitserver
#            --privileged -t -i  --gpus all \
docker run --runtime nvidia \
            -e REPO_HOME=$REPO_HOME \
            -e REPO_WORKING_DIR=$REPO_WORKING_DIR \
            -e REPO_DOCKERFILES=$REPO_DOCKERFILES \
            -e VSCODE_START_DIR=$REPO_HOME \
            -e username=alc \
            -e REGISTRY_ADDR=172.23.0.1:5001 \
            --name $hname \
            -v $dind_docker_folder:/var/run \
            -v $dind_dockerlib_folder:/var/lib/docker \
            -v $REPO_HOME:$REPO_HOME \
            -v $REPO_WORKING_DIR:$REPO_WORKING_DIR \
            -v $REPO_DOCKERFILES:$REPO_DOCKERFILES \
            -v /aa:/aa \
            -v $ALC_HOME:$ALC_HOME \
            --privileged --gpus all \
            --network alcnet \
            --hostname $hname \
            -it \
            alc_dind