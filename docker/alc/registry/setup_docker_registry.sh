#!/usr/bin/env bash

# Check that required environment
if [[ -z "$ALC_DOCKERFILES" ]]; then
    echo "ALC_DOCKERFILES is not defined. Exiting"
    exit 1
fi

# FIXME: This shouldn't be hardcoded
REGISTRY_HOST="172.23.0.1"
REGISTRY_PORT="5001"
REGISTRY_ADDR=$REGISTRY_HOST:$REGISTRY_PORT
echo "Using registry address of ${REGISTRY_ADDR}"

# Generate key-pair other dockers will use to access fileserver if no pair exists yet
mkdir -p $ALC_DOCKERFILES/certs
mkdir -p $ALC_WORKING_DIR/registry
file=$ALC_DOCKERFILES/certs/domain.key
if [[ ! -e "$file" ]]; then
    echo "Generating docker registry certificate"
    openssl req -newkey rsa:4096 -nodes -sha256 -keyout domain.key   -addext "subjectAltName = DNS:$REGISTRY_HOST"   -x509 -days 36500 -out domain.crt

    cp domain.key $ALC_DOCKERFILES/certs/domain.key
    cp domain.crt $ALC_DOCKERFILES/certs/domain.crt
    sudo mkdir -p /etc/docker/certs.d/$REGISTRY_ADDR/
    sudo cp domain.crt /etc/docker/certs.d/$REGISTRY_ADDR/ca.crt
    sudo cp domain.crt /etc/docker/certs.d/$REGISTRY_ADDR/ca.crt

fi

# Start docker registry and set to always restart
echo "Starting docker registry container"
docker run -d --restart=always --name alc_secure_registry -v $ALC_DOCKERFILES/certs:/certs \
-e REGISTRY_HTTP_ADDR=0.0.0.0:$REGISTRY_PORT \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
-e REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/data \
-p $REGISTRY_PORT:$REGISTRY_PORT \
-v $ALC_WORKING_DIR/registry:/data \
registry:2

# Tag latest version of desired docker images and push to registry
./push_built_images.sh $REGISTRY_ADDR