#!/bin/bash
# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

file=/etc/docker/certs.d/$REGISTRY_ADDR
if [[ ! -e "$file" ]]; then
    mkdir -p /etc/docker/certs.d/$REGISTRY_ADDR
    cp $ALC_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$REGISTRY_ADDR/ca.crt
fi

file=~/.ssh
if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
fi

#file=/run_assurance.sh
#if [[ ! -e "$file" ]]; then
cp -r $ALC_HOME_ORIG/alc_utils/docker/run_assurance.sh /run_assurance.sh
chmod +x /run_assurance.sh
#fi
export PATH=$PATH:/:


#ssh-keyscan -H $GIT_SERVER_IP >> ~/.ssh/known_hosts
git config --global 'user.name' "${ALC_USERNAME}"
git config --global 'user.email' "${ALC_USERNAME}@alc.alc"
code-server --bind-addr 0.0.0.0:8080 --auth password --cert /ssl_certs/test.crt --cert-key /ssl_certs/test.key --disable-telemetry $VSCODE_START_DIR & 

set -ex

RUN_FLUXBOX=${RUN_FLUXBOX:-yes}
RUN_XTERM=${RUN_XTERM:-yes}

case $RUN_FLUXBOX in
  false|no|n|0)
    rm -f /app/conf.d/fluxbox.conf
    ;;
esac

case $RUN_XTERM in
  false|no|n|0)
    rm -f /app/conf.d/xterm.conf
    ;;
esac

exec supervisord -c /app/supervisord.conf
