FROM ubuntu:16.04
RUN apt-get update 
#&& apt-get -y install xpra && apt-get -y --purge autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt install -y curl wget gnupg2 software-properties-common 
RUN curl -s  https://winswitch.org/gpg.asc  | gpg -

RUN apt-add-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.9
RUN apt-get install -y apt-transport-https ca-certificates
RUN wget -q https://xpra.org/gpg-2022.asc -O- | apt-key add -
RUN wget -q https://xpra.org/gpg-2018.asc -O- | apt-key add -
RUN wget -O - http://xpra.org/gpg.asc | apt-key add - &&\
    echo "deb http://xpra.org/ xenial main" > /etc/apt/sources.list.d/xpra.list
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y xpra xvfb xterm

RUN curl -s  https://winswitch.org/gpg.asc  | gpg -
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys AB17C654
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookupop=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -

RUN apt update
RUN apt install ros-kinetic-py-trees-msgs -y
RUN apt install ros-kinetic-py-trees -y
RUN apt install ros-kinetic-py-trees-ros -y
RUN apt install ros-kinetic-rqt-py-trees -y
RUN apt install -q -y   \
    ros-kinetic-py-trees* \
    ros-kinetic-rqt-py-trees \
    ros-kinetic-rviz-visual-tools \
    ros-kinetic-moveit-visual-tools \
    ros-kinetic-moveit-ros-visualization

ENV XPRA_DISPLAY=":100"

ARG XPRA_PORT=10000
ENV XPRA_PORT=$XPRA_PORT
EXPOSE $XPRA_PORT

COPY run_in_xpra /usr/bin/run_in_xpra
