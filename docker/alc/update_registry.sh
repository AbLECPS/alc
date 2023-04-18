#!/bin/bash
# Build script for ALC Toolchain docker images (and any supported simulation environments)

# Get date-time in UTC/Zulu time
DATE=$(date -u +'%Y-%m-%d_%H:%M:%S')
DATE+="Z"

# Write STDOUT/STDERR to file (and console)
mkdir -p ./logs
exec > >(tee -a "logs/build_images_${DATE}.log") 2>&1

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
export ALC_WORKSPACE=$ALC_WORKING_DIR

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
    export ALC_REGISTRY_PORT=5001
fi

if [[ -z "$ALC_REGISTRY_ADDR" ]]; then
    echo "ALC_REGISTRY_ADDR is not defined. Using default."
    export ALC_REGISTRY_ADDR="$ALC_REGISTRY_HOST:$ALC_REGISTRY_PORT"
fi

if [[ -z "$ALC_REGISTRY_DATA" ]]; then
    echo "ALC_REGISTRY_DATA is not defined. Using default."
    export ALC_REGISTRY_DATA=$ALC_WORKING_DIR/.registry
fi

echo "Using environment variables:"
echo "ALC_HOME = ${ALC_HOME}"
echo "ALC_DOCKERFILES = ${ALC_DOCKERFILES}"
echo "ALC_WORKING_DIR = ${ALC_WORKING_DIR}"
echo "ALC_FILESERVER_ROOT = ${ALC_FILESERVER_ROOT}"
echo "CURRENT_UID = ${CURRENT_UID}"

file=/etc/docker/certs.d/$ALC_REGISTRY_ADDR/ca.crt
if [[ ! -e "$file" ]]; then
   echo "Certificate for docker registry was not created correctly in the alc-host. Please check."
   exit 1
else

    # Take latest version of desired docker images and push to registry
    echo "Pushing images to registry..."

    image_list=("alc:latest"
                "ros:kinetic-ros-core")
    
    for (( i=0; i<${#image_list[*]}; ++i)); do
        image_name=${image_list[$i]}
        echo 'pushing image to local registry - '$image_name
        docker tag ${image_name} ${ALC_REGISTRY_ADDR}/${image_name}
        docker push ${ALC_REGISTRY_ADDR}/${image_name}
        docker rmi  ${ALC_REGISTRY_ADDR}/${image_name}
    done
    echo "Done."
fi

exit 0
