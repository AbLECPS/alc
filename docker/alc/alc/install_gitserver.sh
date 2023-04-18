#!/bin/bash


apt-get install -y \
  openssh-server \
  git

ssh-keygen -A


mkdir /git-server/
mkdir /git-server/keys 
adduser --home /home/git --shell /usr/bin/git-shell  git
echo git:12345 | chpasswd 
mkdir /home/git/.ssh

