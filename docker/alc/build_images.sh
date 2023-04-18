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
export ALC_WEBGME_SRC=$ALC_HOME/webgme/ALC-Dev
export ALC_WORKSPACE=$ALC_WORKING_DIR
export ALC_JUPYTER_WORKDIR=$ALC_WORKING_DIR/jupyter
export ALC_JUPYTER_MATLAB_WORKDIR=$ALC_WORKING_DIR/jupyter_matlab
export ALC_VERIVITAL_HOME=$ALC_HOME/verivital

echo "Using environment variables:"
echo "ALC_HOME = ${ALC_HOME}"
echo "ALC_DOCKERFILES = ${ALC_DOCKERFILES}"
echo "ALC_WORKING_DIR = ${ALC_WORKING_DIR}"
echo "ALC_FILESERVER_ROOT = ${ALC_FILESERVER_ROOT}"
echo "CURRENT_UID = ${CURRENT_UID}"


# Build various docker images (in-order of entry in array)
image_directory=("dind"
                 "alc"
                 )

build_script=("build.sh"
              "build.sh"
              )

for (( i=0; i<${#image_directory[*]}; ++i)); do
    dir_name=${image_directory[$i]}
    script=${build_script[$i]}
    pushd ${dir_name}
    echo "building ${dir_name} docker image"
    ./${script}

    # Check if build script exited cleanly
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Build script (${script}) returned non-zero exit code (${exit_code}) when building ${dir_name} docker image."
        exit ${exit_code}
    else
        echo "built ${dir_name} docker image"
    fi
    popd
done


############pulling images

image_list=("registry:2"
            "jkarlos/git-server-docker:latest"
            "erichough/nfs-server:latest"
            "mongo:3.4.6"
            "nginx:stable"
            "ros:kinetic-ros-core")

for (( i=0; i<${#image_list[*]}; ++i)); do
    image_name=${image_list[$i]}
    echo 'pulling image from server - '$image_name
    docker pull $image_name
done
##################################

docker tag alc:latest alc_codeserver:latest

docker run -d --name alc_registry -v $ALC_DOCKERFILES/certs:/certs \
-e REGISTRY_HTTP_ADDR=0.0.0.0:$ALC_REGISTRY_PORT \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
-e REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/data \
-p $ALC_REGISTRY_PORT:$ALC_REGISTRY_PORT \
-v $ALC_REGISTRY_DATA:/data \
registry:2

# Tag latest version of desired docker images and push to registry
echo "Pushing images to registry..."

for IMAGE in "alc_codeserver"
do
    docker tag ${IMAGE}:latest ${ALC_REGISTRY_ADDR}/${IMAGE}
    docker push ${ALC_REGISTRY_ADDR}/${IMAGE}
done
echo "Done."

docker stop alc_registry
docker rm   alc_registry
exit 0
