#!/usr/bin/env bash
# Tag latest version of desired docker images and push to registry
echo "Pushing images to registry..."
REGISTRY_ADDR=$1
for IMAGE in "catkin_builder" "alc_codeserver"
do
    docker tag ${IMAGE}:latest ${REGISTRY_ADDR}/${IMAGE}
    docker push ${REGISTRY_ADDR}/${IMAGE}
done
echo "Done."