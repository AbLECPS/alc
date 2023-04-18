#!/bin/bash
python3 -m pip install --upgrade pip
pip install tensorflow==2.6.2
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
apt-get update && apt-get install -y lsb-release sudo && apt-get clean all && \
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
apt install -y curl wget
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
DEBIAN_FRONTEND=noninteractive && apt update  && \
apt install -y \
    ros-melodic-ros-base \
    ros-melodic-gazebo* \
    ros-melodic-rqt* \
    apt install -y python3-dev \
    python3-catkin-pkg \
    python3-rosdep \
    python3-catkin-tools
apt update && apt install -y          \
    protobuf-compiler protobuf-c-compiler \
    xvfb                                  


pip3 install -U rospkg defusedxml pymap3d casadi Cython decorator==4.4.2  numpy  networkx==2.2   matplotlib  six==1.12.0  cloudpickle==1.2.1  
apt install libjpeg-dev zlib1g-dev -y
pip3 install pillow scipy==1.2.0 scikit-image  tornado visdom  protobuf  tensorboardX  imutils argparse psutil  noise torch  torchvision 
pip3 install -U scikit-learn 
pip3 install seaborn

DEBIAN_FRONTEND=noninteractive  apt install -q -y   \
    python-yaml         \
    python-tk           \
    nano                \
    htop                \
    cron                \
    socat


curl -s  https://winswitch.org/gpg.asc  | gpg -

apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y apt-transport-https && \
    DEBIAN_FRONTEND=noninteractive  apt-get install -q -y xpra
pip3 install -U open3d-python
pip3 install netifaces opencv-python 
DEBIAN_FRONTEND=noninteractive apt install -q -y   \
     ros-melodic-rviz-visual-tools \
     ros-melodic-moveit-visual-tools \
     ros-melodic-moveit-ros-visualization

DEBIAN_FRONTEND=noninteractive apt install -q -y   \
    ros-melodic-py-trees* \
    ros-melodic-rqt-py-trees 

pip3 install pydot graphviz

rosdep init
rosdep update
pip3 install empy wstool navpy
# For Py2.7
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
sudo python2.7 get-pip.py
pip2.7 install pymap3d
pip3 install python-dateutil
pip3 install pyyaml
pip3 install gnupg
pip install gnupg
pip install scipy
apt-get install -y git

pip3.6 install docker nbzip jsonlib-python3 textx gitpython addict future http-parser inotify jinja2 pyspark pyxtension webgme_bindings inotify six
wget https://github.com/SchedMD/slurm/archive/refs/tags/slurm-20-02-7-1.zip && \
unzip slurm-20-02-7-1.zip 
mkdir /alc/alc_utils /alc/assurancecasetools /alc/resonate

git clone https://github.com/benmcollins/libjwt.git
git clone https://github.com/nodejs/http-parser.git

