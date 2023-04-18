#!/bin/bash

export VSCODE_DOCKER_IMAGE=alc:latest

if [[ "$(docker images -q $VSCODE_DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
  # get from registry
  echo 'pull docker from registry'
  docker pull $REGISTRY_ADDR/$VSCODE_DOCKER_IMAGE
  echo 'retag docker'
  docker tag $REGISTRY_ADDR/$VSCODE_DOCKER_IMAGE $VSCODE_DOCKER_IMAGE
  docker tag $REGISTRY_ADDR/$VSCODE_DOCKER_IMAGE alc_codeserver:latest
  echo 'done retagging docker'
else
  echo "image exists"
fi

