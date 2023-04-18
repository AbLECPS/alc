#!/bin/bash

file=/etc/docker/certs.d/$REGISTRY_ADDR
if [[ ! -e "$file" ]]; then
    mkdir -p /etc/docker/certs.d/$REGISTRY_ADDR
    cp $REPO_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$REGISTRY_ADDR/ca.crt
fi

file=~/.ssh
if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
fi



if id -u "$username" >/dev/null 2>&1; then
    echo 'user exists. skipping user setup'
else
    echo 'setting up user'
    mkdir -p /home/$username
    groupadd -g 1000 $username && useradd -u 1000 -g 1000 $username 
    chown -R $username:$username /home/$username
    printf "$username     ALL=(ALL) NOPASSWD: ALL\nroot     ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers
    usermod -a -G docker $username
fi


#echo 'starting long sleep'
#sleep 10
# echo 'trying to kill dockerd'
# pid1=$(pgrep dockerd)
# echo $pid1
# kill $pid1
# echo 'waiting for dockerd to die'
# wait $pid1

# echo 'trying to kill containerd'
# pid2=$(pgrep containerd)
# echo $pid2
# kill $pid2
# echo 'waiting for containerd to die'
# wait $pid2

# pkill dockerd
# pkill containerd

# while pgrep -x dockerd > /dev/null; do echo 'trying to kill dockerd'; pkill dockerd; sleep 1; done
# while pgrep -x containerd > /dev/null; do echo 'trying to kill containerd'; pkill containerd; sleep 1; done
# while pgrep -x libcontainerd > /dev/null; do echo 'trying to kill libcontainerd'; pkill libcontainerd; sleep 1; done

if pgrep -x "dockerd" > /dev/null 2>&1; then
    echo ' trying to kill dockerd again'
    pkill dockerd
fi
if pgrep -x "containerd" > /dev/null 2>&1; then
    echo ' trying to kill containerd again'
    pkill containerd
fi

rm -rf /var/run/docker.pid
rm -rf /var/run/docker/containerd/containerd.pid
rm -rf /var/run/docker.sock

# echo 'stopping docker daemon'
# pkill dockerd
# pkill docker-containerd
# rm -rf /var/run/docker.pid
# ps -aefw | grep dockerd

#PID=`ps -ef | grep containerd 'awk {print $2}'`
# echo 'containerd pid is '
# echo $PID

# sleep 60

echo 'starting docker daemon'
dockerd&
echo 'started docker daemon'
echo 'waiting 5 sec for docker daemon to setup'
sleep 5
cd /vslaunch
export ALC_DOCKER_NETWORK=alcnet
export DOCKER_NETWORK_DRIVER=bridge
export ALC_HOME=$ALC_HOME
export REPO_HOME=$REPO_HOME
export ALC_WORKING_DIR=$REPO_WORKING_DIR
export ALC_DOCKERFILES=$REPO_DOCKERFILES
export ALC_SRC=$ALC_HOME
echo $PWD
export
echo 'creating docker network'
docker network create alcnet  --subnet 172.24.0.0/24 --gateway 172.24.0.1
echo 'starting docker-compose...'
echo 'docker images'
docker images
./docker_setup_image.sh
mkdir /aa
docker-compose up 
tail -f /dev/null

