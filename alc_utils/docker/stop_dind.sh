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

docker stop $hname
docker rm $hname
sudo rm -rf $dind_docker_folder

