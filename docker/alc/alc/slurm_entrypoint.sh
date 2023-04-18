#!/usr/bin/env bash
# Adapted from: https://github.com/giovtorres/slurm-docker-cluster/blob/master/docker-entrypoint.sh
set -e

set_munge_key_permissions() {
  echo "---> Setting MUNGE key user-priviliges"
  chown -R alc:alc /etc/munge
  chmod -R 700 /etc/munge/
}

wait_for_munge_key() {
  echo "---> Waiting for MUNGE key in /etc/munge/munge.key ..."
  while [ ! -f /etc/munge/munge.key ]; do sleep 1; done
  echo "---> MUNGE Key available."

  set_munge_key_permissions
}

create_munge_key() {
  # Create MUNGE key if it does not already exist
  file=/etc/munge/munge.key
  if [[ ! -e "$file" ]]; then
    echo "---> Generating new MUNGE Authentication key..."
    /usr/sbin/create-munge-key
  else
    echo "---> Found existing MUNGE Authentication key."
  fi

  set_munge_key_permissions
}

create_jwt_key() {
  # Create JWT RSA key if it does not already exist
  file=/var/slurm/spool/jwt_hs256.key
  if [[ ! -e "$file" ]]; then
    echo "---> Generating new JWT Authentication key..."
    openssl genrsa -out $file 2048
#    chown slurm $file
#    chmod 0700 $file
  else
    echo "---> Found existing JWT Authentication key."
  fi
}

echo "---> Mounting any NFS drives as specified by environment variables..."
/usr/local/bin/mount_nfs.sh

if [ "$1" = "slurmctld" ]; then
  create_munge_key
  create_jwt_key
  file=~/.ssh
  if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
  fi

  echo "---> Starting the MUNGE Authentication service (munged) ..."
  sudo -u alc /usr/sbin/munged &

  echo "---> Starting the Slurm Controller Daemon (slurmctld) ..."
  #  /usr/sbin/slurmctld -Dvvv
  /usr/sbin/slurmctld -D &

  # Wait for all children to finish
  wait
fi

if [ "$1" = "slurmd" ]; then
  wait_for_munge_key
  file=~/.ssh
  if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
  fi

  file=/etc/docker/certs.d/$ALC_REGISTRY_ADDR
  if [[ ! -e "$file" ]]; then
    mkdir -p /etc/docker/certs.d/$ALC_REGISTRY_ADDR
    cp $ALC_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$ALC_REGISTRY_ADDR/ca.crt
  fi

  echo "---> installing  alc utils..."
  cp -r $ALC_HOME/alc_utils/* /alc/alc_utils/.
  #pip3.6 install  /alc/alc_utils/assurancecasetools
  #pip install  /alc/alc_utils/resonate 



  echo "---> Starting the MUNGE Authentication service (munged) ..."
  sudo -u alc /usr/sbin/munged &

  echo "---> Waiting for slurmctld to become active before starting slurmd..."
  until 2>/dev/null >/dev/tcp/alc/6817; do
    echo "-- slurmctld is not available.  Sleeping ..."
    sleep 2
  done
  echo "-- slurmctld is now active ..."

  echo "---> Starting the Slurm Node Daemon (slurmd) ..."
  #  /usr/sbin/slurmd -Dvvv
  /usr/sbin/slurmd -D &

  # Wait for all children to finish
  wait
fi

if [ "$1" = "webgme" ]; then
  create_munge_key
  create_jwt_key
  file=~/.ssh
  if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
    cat $ALC_DOCKERFILES/keys/gitkey.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
  fi

  file=/etc/docker/certs.d/$ALC_REGISTRY_ADDR
  if [[ ! -e "$file" ]]; then
    mkdir -p /etc/docker/certs.d/$ALC_REGISTRY_ADDR
    cp $ALC_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$ALC_REGISTRY_ADDR/ca.crt
  fi

  file=/root/run_assurance_gen.sh
  if [[ ! -e "$file" ]]; then
    cp $ALC_HOME/alc_utils/docker/run_assurance_gen.sh /root/.
    chmod +x /root/run_assurance_gen.sh
  fi

 
  echo "---> installing  alc utils..."
  cp -r $ALC_HOME/alc_utils/* /alc/alc_utils/.
  #pip3.6 install  /alc/alc_utils/assurancecasetools
  #pip install  /alc/alc_utils/resonate 

  

  

  echo "---> Starting sshd ----"
  sudo service ssh restart

  echo "---> Starting the MUNGE Authentication service (munged) ..."
  sudo -u alc /usr/sbin/munged &

  # Launch Slurmctld as daemon and echo logs to STDOUT
  echo "---> Starting the Slurm Controller Daemon (slurmctld) ..."
  #  sudo -u slurm /usr/sbin/slurmctld
  /usr/sbin/slurmctld -D &
  python3.6 /alc/webgme/automate/gradle/src/main/python/slurmrestd_monitor.py &

  echo "---> Starting WebGME ..."
  
  /bin/bash /alc/webgme/automate/gradle/src/main/shell/startup.sh &

  python3.6 "$ALC_HOME"/alc_utils/update_job_status_daemon.py &

  # Wait for all children to finish
  wait
fi

exec "$@"
