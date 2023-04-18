#!/bin/bash

# docker run -it \
#            --rm \
#            --runtime=nvidia \
#            --name bluerov_uuv \
#            --entrypoint=/bin/bash \
#            --network ros \
#            --ip 172.18.0.5 \
#            --hostname aa_iver \
#            --add-host $(hostname):172.18.0.1 \
#            -v $PWD/results:/mnt/results \
#            -v $PWD:/aa \
#            -v $ALC_HOME:$ALC_HOME\
#            -v $ALC_WORKING_DIR:$ALC_WORKING_DIR\
#            -e ROS_MASTER_URI=http://ros-master:11311 \
#            -e ROS_HOSTNAME=aa_iver \
#            -e ALC_SRC=$ALC_HOME \
#            -e ALC_HOME=$ALC_HOME \
#            -e ALC_WORKING_DIR=$ALC_WORKING_DIR \
#            -e PYTHONPATH=$PYTHONPATH:$ALC_SRC \
#            -w /aa \
#            bluerov_sim

docker exec -it bluerov_sim bash

#and then source run_xvfb.sh
