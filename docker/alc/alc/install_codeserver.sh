#!/bin/bash

apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
source $HOME/.nvm/nvm.sh
nvm use 12
curl -fsSL https://code-server.dev/install.sh | sh -s -- --version 3.10.2



printf "slurm     ALL=(ALL) NOPASSWD: ALL\nroot     ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers
printf "alc     ALL=(ALL) NOPASSWD: ALL\nroot     ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers
mkdir /home/alc && chown -R alc:alc /home/alc

usermod -aG docker alc && newgrp docker
groups alc

set -ex; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      bash \
      fluxbox \
      git \
      net-tools \
      novnc \
      supervisor \
      x11vnc \
      xterm \
      xvfb

nvm use 8
