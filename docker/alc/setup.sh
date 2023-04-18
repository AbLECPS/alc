#!/bin/bash
# Setup script for ALC Toolchain docker installation

copy_protected_region_file(){
  # Function to copy file updates while preserving protected regions
  # Assumes protected regions are delimited with "### USER PROTECTED REGION ###" and "### END USER PROTECTED REGION ###"
  # Assumes two arguments: $1 is new file with updates, and $2 is original file with user changes

  # Name arguments and constants
  local new_file=$1
  local orig_file=$2
  local start_region_delimiter="### USER PROTECTED REGION ###"
  local end_region_delimiter="### END USER PROTECTED REGION ###"

  # Check if new file exists
  if [ ! -f "${new_file}" ]; then
    echo "ERROR: ${new_file} does not exist."
    return 1
  fi

  # Check if original file exists
  if [ ! -f "${orig_file}" ]; then
    echo "${orig_file} does not exist. Copying new file."
    cp ${new_file} ${orig_file}
    return 0
  fi

  # Create backup of original file
  local bak_file="${orig_file}.bak"
  echo "Creating backup of existing file $orig_file at $bak_file."
  cp ${orig_file} ${bak_file}

  # Find line numbers for start/end of protected regions in each file
  local orig_region_start=$(grep -n "$start_region_delimiter" $orig_file | cut -d : -f1)
  local orig_region_end=$(grep -n "$end_region_delimiter" $orig_file | cut -d : -f1)
  local new_region_start=$(grep -n "$start_region_delimiter" $new_file | cut -d : -f1)
  local new_region_end=$(grep -n "$end_region_delimiter" $new_file | cut -d : -f1)

  if [ -n "${orig_region_start}" ]; then
    echo "Found user protected region in ${orig_file}"

    # Read and store user protected text
    local user_text=$(sed -n "${orig_region_start},${orig_region_end}p" ${orig_file})

    # Delete the generic protected region from the new file and send output to overwrite original file.
    # Insert stored user text where generic protected region was previously located.
    echo "Updating existing file ${orig_file} with new file ${new_file}."
    local insert_line=$(( ${new_region_start} - "1" ))
    sed -e "${new_region_start},${new_region_end}d" ${new_file} > ${orig_file}
    sed -i "${insert_line}r"<(echo "$user_text") ${orig_file}
  else
    # No user protected region exits. Overwrite entire file
    echo "WARNING: No user protected region found in ${orig_file}."
    echo "Will overwrite entire file, any previous changes can be recovered from ${bak_file}."
    cp ${new_file} ${orig_file}
  fi

  return 0
}


# Get any named arguments. Restore positional arguments when done.
POSITIONAL=()
echo "Got arguments: "
while [[ $# -gt 0 ]]
do
    key="$1"
    echo "${key}"
    if [[ $key != *--* && $key == *-* ]] 
    then 
	str="${key##*-}"
	len=${#str} 
	if [[ $len > 1 ]] 
    	then 
		echo "ERROR: Single hyphen arguments can only contain one letter."
		exit 1
	fi
    fi
    # Get any arguments used by this script. Other arguments left alone
    case $key in
	  -b|--build)
	    BUILD=1
	    ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      ;;
    esac
    shift
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Get date-time in UTC/Zulu time
DATE=$(date -u +'%Y-%m-%d_%H:%M:%S')
DATE+="Z"

# Write STDOUT/STDERR to file (in addition to console)
exec > >(tee -a "logs/setup_${DATE}.log") 2>&1

# Check that required environment variables are defined
if [[ -z "$ALC_HOME" ]]; then
    echo "ALC_HOME is not defined. Exiting."
    exit 1
fi
if [[ -z "$ALC_WORKING_DIR" ]]; then
    echo "ALC_WORKING_DIR is not defined. Exiting."
    exit 1
fi
if [[ -z "$ALC_DOCKERFILES" ]]; then
    echo "ALC_DOCKERFILES is not defined. Exiting"
    exit 1
fi

# Define any derived environment variables
export ALC_SRC=$ALC_HOME
export ALC_WEBGME_SRC=$ALC_HOME/webgme/
export ALC_WORKSPACE=$ALC_WORKING_DIR
export ALC_JUPYTER_WORKDIR=$ALC_WORKING_DIR/jupyter
export ALC_JUPYTER_MATLAB_WORKDIR=$ALC_WORKING_DIR/jupyter_matlab
export ALC_VERIVITAL_HOME=$ALC_HOME/verivital
export ALC_GITSERVER_ROOT=$ALC_WORKING_DIR/.gitserver
export ALC_REGISTRY_DATA=$ALC_WORKING_DIR/.registry
export ALC_DOCKER_NETWORK="alcnet"
export ALC_DOCKER_NETWORK_GATEWAY="172.23.0.1"
export ALC_DOCKER_NETWORK_SUBNET="172.23.0.0/24"
export ALC_REGISTRY_HOST=$ALC_DOCKER_NETWORK_GATEWAY
export ALC_REGISTRY_PORT=5001
export ALC_GITSERVER_HOST=$ALC_DOCKER_NETWORK_GATEWAY
export ALC_GITSERVER_PORT=2222
export ALC_SSH_HOST=$ALC_DOCKER_NETWORK_GATEWAY
export ALC_SSH_PORT=5222
export ALC_GITSERVER_URL="$ALC_DOCKER_NETWORK_GATEWAY:$ALC_GITSERVER_PORT"
export ALC_REGISTRY_ADDR="$ALC_REGISTRY_HOST:$ALC_REGISTRY_PORT"


echo "Using environment variables:"
echo "ALC_HOME = ${ALC_HOME}"
echo "ALC_DOCKERFILES = ${ALC_DOCKERFILES}"
echo "ALC_WORKING_DIR = ${ALC_WORKING_DIR}"
echo "ALC_FILESERVER_ROOT = ${ALC_FILESERVER_ROOT}"
echo "ALC_GITSERVER_ROOT = ${ALC_GITSERVER_ROOT}"
echo "ALC_REGISTRY_DATA = ${ALC_REGISTRY_DATA}"
echo "CURRENT_UID = ${CURRENT_UID}"

# Give current user ownership permissions where required
# Ownership is returned to alc user at end of script
sudo chown -R $USER:$USER $ALC_WORKING_DIR $ALC_DOCKERFILES

# Make sure all required directories exist
mkdir -p $ALC_WORKSPACE/{cache,ros,.docker,.registry,.gitserver,.build,.exec,jupyter,jupyter_matlab} \
         $ALC_DOCKERFILES/{ssl_certs,certs,logs,token_keys,keys,gitkey,sshcontents,blob-local-storage,db,config} \
         $ALC_DOCKERFILES/worker/data/{db,blob} \
         $ALC_DOCKERFILES/worker/worker-cache \
         $ALC_DOCKERFILES/slurm/{etc,munge} \
         $ALC_DOCKERFILES/slurm/var/{log,spool,run} \
         $ALC_DOCKERFILES/nginx/{etc,logs}

mkdir -p $ALC_GITSERVER_ROOT/{keys,repos}
mkdir -p $ALC_WORKSPACE/.docker/.users 

find $ALC_WORKSPACE

find $ALC_DOCKERFILES
# Copying webgme sources to build docker
#rsync -av --progress $ALC_WEBGME_SRC/* ./webgme --exclude node_modules --exclude blob-local-storage

# Copying template files to jupyter work dir
echo "Copying jupyter notebook init files to workspace..."
cp $ALC_SRC/docker/alc/jupyter/initn* $ALC_JUPYTER_WORKDIR/.
cp $ALC_SRC/docker/alc/jupyter/resultnb $ALC_JUPYTER_WORKDIR/.
cp $ALC_SRC/docker/alc/jupyter_matlab/initmatlabnb $ALC_JUPYTER_MATLAB_WORKDIR/.
echo "Done copying jupyter notebook init files."

# Copy SLURM related files to ALC Dockerfiles if they do not already exist.
pushd $ALC_DOCKERFILES/slurm #|| { echo "ERROR: pushd into ${ALC_DOCKERFILES}/slurm failed!"; exit 1 }
# Initialize any other directories that need to be persisted between docker startup/shutdown
touch $ALC_DOCKERFILES/slurm/var/log/{slurmctld,slurmd,jobcomp}.log \
      $ALC_DOCKERFILES/slurm/var/spool/{node_state,front_end_state,job_state,assoc_mgr_state,assoc_usage,qos_usage,fed_mgr_state}

# Copy slurm configuration files while preserving existing user edits
echo "Copying slurm configuration files to ${ALC_DOCKERFILES}/slurm"
copy_protected_region_file $ALC_HOME/docker/alc/slurm/etc/slurm.conf $ALC_DOCKERFILES/slurm/etc/slurm.conf
copy_protected_region_file $ALC_HOME/docker/alc/slurm/etc/gres.conf $ALC_DOCKERFILES/slurm/etc/gres.conf
cp $ALC_HOME/docker/alc/slurm/etc/slurmrestd.conf $ALC_DOCKERFILES/slurm/etc/slurmrestd.conf
popd #|| { echo "ERROR: popd from ${ALC_DOCKERFILES}/slurm failed!"; exit 1 }

# Copy NGINX config files
echo "Copying NGINX config files to ${ALC_DOCKERFILES}/nginx/etc"
cp $ALC_HOME/docker/alc/nginx/nginx.conf $ALC_DOCKERFILES/nginx/etc/nginx.conf
cp $ALC_HOME/docker/alc/nginx/mime.types $ALC_DOCKERFILES/nginx/etc/mime.types

# FIXME: Also add alc user to group 'docker'. Is this required?
# Create new group & user for ALC services if they do not exist
echo "Creating group 'alc' and user account 'alc' if they do not already exist. This will require 'sudo' authentication."
getent group alc || sudo groupadd -g 10181 alc
id -u alc || sudo useradd -r -u 10181 -g 10181 alc
sudo usermod -G docker alc

##Registry settings


echo "Using registry address of ${ALC_REGISTRY_ADDR}"

# Generate key-pair other dockers will use to access fileserver if no pair exists yet
file=$ALC_DOCKERFILES/certs/domain.key
if [[ ! -e "$file" ]]; then
    echo "Generating docker registry certificate"
    openssl req -newkey rsa:4096 -nodes -sha256 -keyout $ALC_DOCKERFILES/certs/domain.key  -addext "subjectAltName = IP:$ALC_REGISTRY_HOST" -subj "/C=US/ST=TN/L=Nashville/O=ALC/CN=ALC"  -x509 -days 36500 -out $ALC_DOCKERFILES/certs/domain.crt
    sudo mkdir -p /etc/docker/certs.d/$ALC_REGISTRY_ADDR/
    sudo cp $ALC_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$ALC_REGISTRY_ADDR/ca.crt
    sudo cp $ALC_DOCKERFILES/certs/domain.crt /etc/docker/certs.d/$ALC_REGISTRY_ADDR/ca.crt
fi
if [[ ! -e "$file" ]]; then
   echo "openssl failed.. Please check your openssl version. Update to atleast 1.1.1."
   exit 1
fi

#recreate docker network
echo "Recreating docker network"
docker network rm $ALC_DOCKER_NETWORK
docker network create $ALC_DOCKER_NETWORK --subnet $ALC_DOCKER_NETWORK_SUBNET  --gateway $ALC_DOCKER_NETWORK_GATEWAY

# Generate key-pair 
file=$ALC_DOCKERFILES/ssl_certs/test.key
if [[ ! -e "$file" ]]; then
  echo "Generating ssl_certs"
  openssl req -x509 -nodes -days 36500 -newkey rsa:2048 -subj "/C=US/ST=TN/L=Nashville/O=ALC/CN=ALC" -keyout $ALC_DOCKERFILES/ssl_certs/test.key -out $ALC_DOCKERFILES/ssl_certs/test.crt
fi

if [[ ! -e "$file" ]]; then
   echo "openssl failed.. Please check your openssl version. Update to atleast 1.1.1."
   exit 1
fi

file=$ALC_DOCKERFILES/keys/gitkey.pub
if [[ ! -e "$file" ]]; then
  echo "Generating keys for git"
  ssh-keygen -t rsa -f $ALC_DOCKERFILES/keys/gitkey -q -N ""

  echo "Generating ssh contents"
  cp $ALC_DOCKERFILES/keys/gitkey.pub $ALC_GITSERVER_ROOT/keys/.
  cp $ALC_DOCKERFILES/keys/gitkey $ALC_DOCKERFILES/sshcontents/.
  echo "Host *" > $ALC_DOCKERFILES/sshcontents/config     
  echo " StrictHostKeyChecking no" >> $ALC_DOCKERFILES/sshcontents/config
  echo -e "Host $ALC_GITSERVER_HOST\n  Hostname $ALC_GITSERVER_HOST\n  PreferredAuthentications publickey\n  IdentityFile ~/.ssh/gitkey" >> $ALC_DOCKERFILES/sshcontents/config
  chmod 600 $ALC_DOCKERFILES/sshcontents/gitkey

  echo "Starting git server to configure ssh"
  docker run -d --name alc_gitserver \
  -p $ALC_GITSERVER_PORT:22 \
  -v $ALC_GITSERVER_ROOT/keys:/git-server/keys \
  -v $ALC_GITSERVER_ROOT/repos:/git-server/repos \
  --network $ALC_DOCKER_NETWORK \
  jkarlos/git-server-docker

  sleep 5

  echo "Running ssh-keyscan"
  ssh-keyscan  -p $ALC_GITSERVER_PORT  $ALC_GITSERVER_HOST > $ALC_DOCKERFILES/sshcontents/known_hosts


  echo "Stopping gitserver"
  docker stop alc_gitserver
  docker rm alc_gitserver
fi



echo "git-repo initialization"
### create bluerov.git
export repo_name='bluerov'
file=$ALC_GITSERVER_ROOT/repos/${repo_name}.git
if [[ ! -e "$file" ]]; then
  mkdir -p $ALC_GITSERVER_ROOT/repos/temp$repo_name
  pushd $ALC_GITSERVER_ROOT/repos/temp$repo_name
  cp -r $ALC_HOME/alc_ros .
  cp -r $ALC_HOME/alc_utils .
  find . -type d -name ".git" -exec rm -rf {} +
  git init --shared=true
  git add -A
  #git commit -c user.name='admin' -c user.email='admin@alc' -m "initial setup"
  git config --local user.name  "admin"
  git config --local user.email  "admin@alc.alc"
  git commit -m "initial setup"
  popd
  pushd $ALC_GITSERVER_ROOT/repos
  git clone --bare temp$repo_name $repo_name.git
  rm -rf temp$repo_name
  popd
fi
#####################




############################# NFS Setup ###############################
# Install NFS server kernel module on host
echo "setting up nfs modules"
sudo apt-get install -y nfs-kernel-server

# Copy NFS related files to ALC Dockerfiles if they do not already exist
echo "INFO: Preparing NFS fileserver configuration files..."
mkdir -p $ALC_DOCKERFILES/nfs/etc
pushd $ALC_DOCKERFILES/nfs #|| { echo "ERROR: pushd into ${ALC_DOCKERFILES}/slurm failed!"; exit 1 }
file=./etc/exports
if [[ ! -e "$file" ]]; then
  echo "Copying file system exports file for NFS server to ${ALC_DOCKERFILES}/nfs"
  cp $ALC_HOME/docker/alc/slurm/nfs/etc/exports $ALC_DOCKERFILES/nfs/etc/.
else
  echo  "NFS exports file already exists at ${ALC_DOCKERFILES}/nfs/etc/exports. Will not overwrite."
fi
popd #|| { echo "ERROR: popd from ${ALC_DOCKERFILES}/slurm failed!"; exit 1 }
########################### End NFS Setup ##############################


# Pull ros-core docker.
# This should be done automatically by docker, but fails on some machines for unknown reason.
echo "Pulling ROS Kinetic docker base image..."
docker pull ros:kinetic-ros-core

# Build docker images if desired
if [ -z "$BUILD" ]; then
    echo "Skipping build of docker images.";
else
    echo "Building docker images..."
    ./build_images.sh
    # Check if build script exited cleanly
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
  echo "############################# BUILD FAILED ######################################"
  echo "ERROR: Build script returned non-zero exit code (${exit_code}). See output above for detailed error information."
  echo "#################################################################################"
  exit ${exit_code}
    fi
    echo "Done building images."
fi

# Creating token-keys for webgme
pushd $ALC_DOCKERFILES/token_keys
file=./private_key
if [ ! -e "$file" ]; then
  echo "Generating WebGME token keys."
  openssl genrsa -out private_key 1024
  openssl rsa -in private_key -pubout > public_key
else
  echo "WebGME token keys already exist. Skipping key generation."
fi
popd

if [ -z "$BUILD" ]; then
	    echo "Skipping build of docker documentation images.";
else
	# Build documentation
	pushd $ALC_HOME/doc
	echo "Building documentation..."
	./build.sh
	exit_code=$?
	if [ $exit_code -ne 0 ]; then
	  echo "############################# BUILD FAILED ######################################"
	  echo "WARNING: Documentation build script returned non-zero exit code (${exit_code}). See output above for detailed error information."
	  echo "#################################################################################"
	fi
	echo "Done building documentation."
	popd
fi

if [ -z "$BUILD" ]; then
  echo "Skipping build of catkin builder workspace.";
else
	# Build ROS package workspaces
	pushd catkin_builder
	echo "Building ROS package workspaces..."
	./build_workspaces.sh
	exit_code=$?
	if [ $exit_code -ne 0 ]; then
	  echo "############################# BUILD FAILED ######################################"
	  echo "ERROR: ROS packages build script returned non-zero exit code (${exit_code}). See output above for detailed error information."
	  echo "#################################################################################"
	  exit ${exit_code}
	fi
	echo "Done building ROS package workspaces."
	popd
fi

# Set ownership and permissions of ALC directories appropriately.
# ALC user and group given Read, Write, Execute permissions to all directories.
# ALC user and group added Read and Write permissions to all files. Execution permissions are not modified for files.
echo "Setting ownership and permission levels for ALC directories..."
sudo chown -R alc:alc $ALC_WORKING_DIR $ALC_DOCKERFILES
sudo find $ALC_WORKING_DIR -type d -exec chmod 775 {} +
sudo find $ALC_DOCKERFILES -type d -exec chmod 775 {} +
sudo find $ALC_WORKING_DIR -type f -exec chmod ug+rw {} +
sudo find $ALC_DOCKERFILES -type f -exec chmod ug+rw {} +
echo "Done setting ownership and permission levels."

# Create SQL database volume
docker volume create alc_sql_db
