#!/bin/sh
docker build -t alc_1:latest -f Dockerfile.bluerov .
export ALC_WEBGME_SRC=$ALC_HOME/webgme/
rsync -av --progress $ALC_WEBGME_SRC/* . --exclude node_modules --exclude blob-local-storage
docker build -t alc_2:latest -f Dockerfile.webgme .
rm -rf ALC-Dev
docker build -t alc_3:latest -f Dockerfile.codeserver .
docker build -t alc_4:latest -f Dockerfile.codeserver .
docker tag alc_4:latest alc:latest
docker rmi alc_1:latest alc_2:latest alc_3:latest alc_4:latest