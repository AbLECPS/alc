#!/bin/bash
set -e

pids=""

DOCKER_PRE_OPTIONS=()
POSITIONAL=()
DOCKER_COMPOSE_CMD="up"
TARGET="services"
BUILD=false
DOCKER_NETWORK_DRIVER="bridge"
while [[ $# -gt 0 ]]
do
    key="$1"
    # Get any arguments used by this script. Other arguments will be passed on to docker-compose
    case $key in
        -p|--project-name)
        DOCKER_PRE_OPTIONS+=("$1")
        DOCKER_PRE_OPTIONS+=("$2")
        shift # past value
        ;;
        --worker)
        DOCKER_PRE_OPTIONS+=("-f")
        DOCKER_PRE_OPTIONS+=("docker-compose-worker.yml")
        TARGET="worker"
        ;;
        -b|--build)
        BUILD=true
        POSITIONAL+=("--build")
        ;;
        --down)
        DOCKER_COMPOSE_CMD="down"
        ;;
        --overlay)
        DOCKER_NETWORK_DRIVER="overlay"
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        ;;
    esac
    shift
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Argument compatability checks
if [[ "$TARGET" == "worker" ]] && [[ "$DOCKER_NETWORK_DRIVER" != "overlay" ]]; then
  echo "WARNING: 'worker' option typically requires 'overlay' docker network driver, but a different driver ('${DOCKER_NETWORK_DRIVER}') was selected. Proceed with caution."

fi

# Define cleanup function
function cleanup() {
  docker-compose "${DOCKER_PRE_OPTIONS[@]}" down
  exit $?
}

# If Ctrl+C is pressed (SIGINT) or SIGTERM is received, run cleanup function
trap cleanup SIGINT SIGTERM

# Get date-time in UTC/Zulu time
DATE=$(date -u +'%Y-%m-%d_%H:%M:%S')
DATE+="Z"

# Check that all necessary environment parameters are set
if [[ -z "$ALC_HOME" ]]; then
    echo "ALC_HOME is not defined. Exiting."
    exit 1
fi
if [[ -z "$ALC_WORKING_DIR" ]]; then
    echo "ALC_WORKING_DIR is not defined. Exiting."
    exit 1
fi
if [[ -z "$ALC_DOCKERFILES" ]]; then
    echo "ALC_DOCKERFILES is not defined. Exiting."
    exit 1
fi
if [[ -z "$ALC_PORT" ]]; then
    echo "ALC_PORT is not defined. Using default value of 8000."
    ALC_PORT="8000"
fi
if [[ -z "$ALC_DOCKER_NETWORK" ]]; then
    echo "ALC_DOCKER_NETWORK is not defined. Using default network name."
    ALC_DOCKER_NETWORK="alcnet"
fi
if [[ -z "$ALC_FILESERVER_ROOT" ]]; then
    echo "ALC_FILESERVER_ROOT is not defined. Using default."
    ALC_FILESERVER_ROOT="/tmp/alc"
fi

if [[ -z "$ALC_DOCKER_NETWORK_GATEWAY" ]]; then
    echo "ALC_DOCKER_NETWORK_GATEWAY is not defined. Using default."
    export ALC_DOCKER_NETWORK_GATEWAY="172.23.0.1"
fi

if [[ -z "$ALC_DOCKER_NETWORK_SUBNET" ]]; then
    echo "ALC_DOCKER_NETWORK_SUBNET is not defined. Using default."
    export ALC_DOCKER_NETWORK_SUBNET="172.23.0.0/24"
fi


if [[ -z "$ALC_REGISTRY_HOST" ]]; then
    echo "ALC_REGISTRY_HOST is not defined. Using default."
    export ALC_REGISTRY_HOST=$ALC_DOCKER_NETWORK_GATEWAY
fi

if [[ -z "$ALC_REGISTRY_PORT" ]]; then
    echo "ALC_REGISTRY_PORT is not defined. Using default."
    export ALC_REGISTRY_PORT="5001"
fi

if [[ -z "$ALC_REGISTRY_ADDR" ]]; then
    echo "ALC_REGISTRY_ADDR is not defined. Using default."
    export ALC_REGISTRY_ADDR="$ALC_REGISTRY_HOST:$ALC_REGISTRY_PORT"
fi

if [[ -z "$ALC_REGISTRY_DATA" ]]; then
    echo "ALC_REGISTRY_DATA is not defined. Using default."
    export ALC_REGISTRY_DATA=$ALC_WORKING_DIR/.registry
fi

if [[ -z "$ALC_GITSERVER_HOST" ]]; then
    echo "ALC_GITSERVER_HOST is not defined. Using default."
    export ALC_GITSERVER_HOST=$ALC_DOCKER_NETWORK_GATEWAY
fi

if [[ -z "$ALC_GITSERVER_PORT" ]]; then
    echo "ALC_GITSERVER_PORT is not defined. Using default."
    export ALC_GITSERVER_PORT="2222"
fi

if [[ -z "$ALC_GITSERVER_URL" ]]; then
    echo "ALC_GITSERVER_URL is not defined. Using default."
    export ALC_GITSERVER_URL="$ALC_GITSERVER_HOST:$ALC_GITSERVER_PORT"
fi

if [[ -z "$ALC_GITSERVER_ROOT" ]]; then
    echo "ALC_GITSERVER_ROOT is not defined. Using default."
    export ALC_GITSERVER_ROOT=$ALC_WORKING_DIR/.gitserver
fi


if [[ -z "$ALC_SSH_PORT" ]]; then
    echo "ALC_SSH_PORT is not defined. Using default."
    export ALC_SSH_PORT="5222"
fi

if [[ -z "$ALC_SSH_HOST" ]]; then
    echo "ALC_SSH_HOST is not defined. Using default."
    export ALC_SSH_HOST=$ALC_DOCKER_NETWORK_GATEWAY
fi



# Define all environment variables the docker build scripts may need
export ALC_SRC=$ALC_HOME
export ALC_WEBGME_SRC=$ALC_HOME/webgme/ALC-Dev
export ALC_WORKSPACE=$ALC_WORKING_DIR
export ALC_JUPYTER_WORKDIR=$ALC_WORKING_DIR/jupyter
export ALC_JUPYTER_MATLAB_WORKDIR=$ALC_WORKING_DIR/jupyter_matlab
export ALC_FILESERVER_ROOT=$ALC_FILESERVER_ROOT
export ALC_VERIVITAL_HOME=$ALC_HOME/verivital
export ALC_PORT=$ALC_PORT
export ALC_DOCKER_NETWORK=$ALC_DOCKER_NETWORK
export DOCKER_NETWORK_DRIVER=$DOCKER_NETWORK_DRIVER
export CURRENT_USER=$(id -u):$(id -g)
export ALC_USER=$(id alc -u):$(id alc -g)
export DOCKER_GID=$(getent group docker | awk -F: '{printf "%s", $3}')




echo "Using environment variables:"
echo "ALC_HOME = ${ALC_HOME}"
echo "ALC_DOCKERFILES = ${ALC_DOCKERFILES}"
echo "ALC_WORKING_DIR = ${ALC_WORKING_DIR}"
echo "ALC_FILESERVER_ROOT = ${ALC_FILESERVER_ROOT}"
echo "CURRENT_USER = ${CURRENT_USER}"
echo "ALC_USER = ${ALC_USER}"
echo "DOCKER_GID = ${DOCKER_GID}"
echo "ALC_DOCKER_NETWORK_GATEWAY = ${ALC_DOCKER_NETWORK_GATEWAY}"
echo "ALC_DOCKER_NETWORK_SUBNET = ${ALC_DOCKER_NETWORK_SUBNET}"
echo "ALC_REGISTRY_ADDR = ${ALC_REGISTRY_ADDR}"
echo "ALC_REGISTRY_DATA = ${ALC_REGISTRY_DATA}"
echo "ALC_GITSERVER_URL = ${ALC_GITSERVER_URL}"
echo "ALC_GITSERVER_ROOT = ${ALC_GITSERVER_ROOT}"

# Create docker network, if it does not already exist
if [[ "$DOCKER_NETWORK_DRIVER" == "bridge" ]]; then
  docker network inspect $ALC_DOCKER_NETWORK >/dev/null 2>&1 || \
  docker network create --driver $DOCKER_NETWORK_DRIVER --subnet $ALC_DOCKER_NETWORK_SUBNET  --gateway $ALC_DOCKER_NETWORK_GATEWAY  $ALC_DOCKER_NETWORK
elif [[ "$DOCKER_NETWORK_DRIVER" == "overlay" ]]; then
  export ALC_DOCKER_NETWORK="${ALC_DOCKER_NETWORK}_overlay"
  # For overlay networks, only need to create if this is the master node (ie. "services" target)
  if [[ "$TARGET" == "services" ]]; then
    docker network inspect $ALC_DOCKER_NETWORK >/dev/null 2>&1 || \
    docker network create --driver $DOCKER_NETWORK_DRIVER --subnet $ALC_DOCKER_NETWORK_SUBNET  --gateway $ALC_DOCKER_NETWORK_GATEWAY --attachable $ALC_DOCKER_NETWORK
  fi
else
  echo "ERROR: Unrecognized docker network driver '${DOCKER_NETWORK_DRIVER}'."
  exit 1
fi

# Setup enviornment for MATLAB
. ${ALC_HOME}/docker/alc/setup_matlab_env.sh

# Check docker version and launch docker-compose with any user-specified options
SERVER_VERSION=$(docker version -f "{{.Server.Version}}")
SERVER_VERSION_MAJOR=$(echo "$SERVER_VERSION"| cut -d'.' -f 1)
SERVER_VERSION_MINOR=$(echo "$SERVER_VERSION"| cut -d'.' -f 2)
SERVER_VERSION_BUILD=$(echo "$SERVER_VERSION"| cut -d'.' -f 3)

# FIXME: Update this when new version of docker compose for >=19.0.3 is available
if [ "${SERVER_VERSION_MAJOR}" -ge 19 ] && \
   [ "${SERVER_VERSION_MINOR}" -ge 0 ]  && \
   [ "${SERVER_VERSION_BUILD}" -ge 3 ]; then
    #echo "Docker version >= 19.0.3. Using standard docker compose file."
    echo "Executing command: 'docker-compose ${DOCKER_PRE_OPTIONS[@]} up ${POSITIONAL[@]}'" > "logs/run_services_${DATE}.log"
    docker-compose "${DOCKER_PRE_OPTIONS[@]}" "${DOCKER_COMPOSE_CMD}" "${POSITIONAL[@]}" |& tee -a "logs/run_services_${DATE}.log"
    pids="$pids $!"
else
    #echo "Docker version less than 19.0.3. Using legacy docker compose file and nvidia-runtime library."
    echo "Executing command: 'docker-compose ${DOCKER_PRE_OPTIONS[@]} up ${POSITIONAL[@]}'" > "logs/run_services_${DATE}.log"
    docker-compose "${DOCKER_PRE_OPTIONS[@]}" "${DOCKER_COMPOSE_CMD}" "${POSITIONAL[@]}" |& tee -a "logs/run_services_${DATE}.log"
    pids="$pids $!"
fi

# Run any tasks which need ALC dockers to be running
if [[ "$BUILD" = true ]] && [[ "$TARGET" == "services" ]]; then
  ./register_cluster.sh
fi

for pid in $pids; do
    wait $pid
done
