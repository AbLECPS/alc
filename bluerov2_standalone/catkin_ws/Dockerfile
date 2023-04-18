# FROM tensorflow/tensorflow:2.5.1-jupyter
# FROM tensorflow/tensorflow:2.5.1-gpu-jupyter
ARG TAG=gpu-jupyter
FROM tensorflow/tensorflow:2.5.1-$TAG

RUN python3 -m pip install --upgrade pip
RUN pip install tensorflow==2.6.2

# ROS Melodic
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y lsb-release sudo && apt-get clean all && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install -y curl wget
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update 
RUN apt install -y \
    ros-melodic-ros-base \
    ros-melodic-gazebo* \
    ros-melodic-rqt* 

RUN apt install -y python3-dev \
    python3-catkin-pkg \
    python3-rosdep \
    python3-catkin-tools

RUN apt update && apt install -y          \
    protobuf-compiler protobuf-c-compiler \
    xvfb                                  


RUN pip3 install -U rospkg defusedxml pymap3d casadi Cython
RUN pip3 install decorator==4.4.2     
RUN pip3 install numpy       
RUN pip3 install networkx==2.2        
RUN pip3 install matplotlib     
RUN pip3 install six==1.12.0          
RUN pip3 install cloudpickle==1.2.1   
RUN apt install libjpeg-dev zlib1g-dev -y
RUN pip3 install pillow         
RUN pip3 install scipy==1.2.0
RUN pip3 install scikit-image  
RUN pip3 install tornado==3.2.1       
RUN pip3 install visdom               
RUN pip3 install protobuf             
RUN pip3 install tensorboardX         
RUN pip3 install imutils              
RUN pip3 install argparse             
RUN pip3 install psutil               
RUN pip3 install noise
RUN pip3 install torch  torchvision 
RUN pip3 install -U scikit-learn
RUN pip3 install seaborn

RUN apt install -q -y   \
    python-yaml         \
    python-tk           \
    nano                \
    htop                \
    cron                \
    socat


RUN curl -s  https://winswitch.org/gpg.asc  | gpg -

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y xpra
RUN pip3 install -U open3d-python
RUN pip3 install netifaces
RUN pip3 install opencv-python

RUN apt install -q -y   \
     ros-melodic-rviz-visual-tools \
     ros-melodic-moveit-visual-tools \
     ros-melodic-moveit-ros-visualization

RUN apt install -q -y   \
    ros-melodic-py-trees* \
    ros-melodic-rqt-py-trees 

RUN pip3 install pydot
RUN pip3 install graphviz

RUN rosdep init
RUN rosdep update
RUN pip3 install empy
RUN pip3 install wstool
RUN pip3 install navpy
# For Py2.7
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN sudo python2.7 get-pip.py
RUN pip2.7 install pymap3d
RUN pip3 install python-dateutil==2.7.0
RUN pip3 install pyyaml
RUN pip3 install gnupg
RUN pip install gnupg
RUN pip install scipy
RUN apt-get install -y git
ADD build_tf2_geometry.sh /.
RUN chmod +x /build_tf2_geometry.sh && /build_tf2_geometry.sh
ADD build_cvbridge.sh /.
RUN chmod +x /build_cvbridge.sh && /build_cvbridge.sh
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN sudo python2.7 get-pip.py

RUN apt install -y ros-melodic-uuv-simulator
RUN sed -i 's/env python2/env python/g' /opt/ros/melodic/lib/uuv_descriptions/spawn_model
RUN sed -i 's/env python2/env python/g' /opt/ros/melodic/lib/uuv_thruster_manager/thruster_allocator.py
RUN sed -i 's/env python2/env python/g' /opt/ros/melodic/lib/uuv_trajectory_control/rov_pid_controller.py
RUN sed -i 's/from thruster import Thruster/from .thruster import Thruster/g' /opt/ros/melodic/lib/python2.7/dist-packages/uuv_thrusters/models/thruster_proportional.py
RUN sed -i 's/import cPickle/import _pickle as cPickle/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/blackboard.py
RUN sed -i 's/print "%s" % sub_blackboard.data/print("%s" % sub_blackboard.data)/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/programs/blackboard_watcher.py
RUN sed -i 's/print "%s" % s/print("%s" % s)/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/programs/blackboard_watcher.py
RUN sed -i 's/Exception, e:/Exception(e):/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/programs/blackboard_watcher.py
RUN sed -i 's/print "%s" % s/print("%s" % s)/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/programs/tree_watcher.py
RUN sed -i 's/print "%s" % msg.data/print("%s" % msg.data)/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/programs/tree_watcher.py
RUN sed -i 's/basestring/str/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/subscribers.py
RUN sed -i 's/iteritems/items/g' /opt/ros/melodic/lib/python2.7/dist-packages/py_trees_ros/subscribers.py
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN pip3 install future pycryptodomex pyserial