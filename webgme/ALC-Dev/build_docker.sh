docker build -t alc_alc:update -f Dockerfile_alc .
docker tag alc_alc:latest alc_alc:past
docker tag alc_alc:update alc_alc:latest