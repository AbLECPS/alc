#From jpetazzo/dind
From daangn/dind:ubuntu-18.04

USER root
RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
    apt-key add - &&\
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) &&\
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
    tee /etc/apt/sources.list.d/nvidia-container-runtime.list &&\
    apt-get update &&\
    apt-get install -y --force-yes nvidia-container-runtime
COPY daemon.json /etc/docker/ 
COPY run_dind.sh /
RUN chmod +x /run_dind.sh
RUN curl -L https://github.com/docker/compose/releases/download/1.29.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose
COPY docker-compose.yml /vslaunch/
COPY docker_setup_image.sh /vslaunch/
RUN chmod +x /vslaunch/docker_setup_image.sh
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Remove any existing docker packages, then install latest version.
RUN apt-get remove docker docker-engine docker.io containerd runc
RUN apt-get update
RUN apt upgrade -y
RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

#RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
#RUN apt-key fingerprint 0EBFCD88
#RUN add-apt-repository \
#   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
#   $(lsb_release -cs) \
#   stable"
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update
RUN apt-get install -y docker-ce docker-ce-cli containerd.io
EXPOSE 8080
EXPOSE 6901
EXPOSE 6900

ENV username alcuser
ENTRYPOINT ["/run_dind.sh"]