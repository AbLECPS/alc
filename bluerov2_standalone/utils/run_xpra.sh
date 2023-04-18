#!/bin/bash
pushd $ALC_HOME/catkin_ws
. setup_gazebo_ros_connection.sh
. $ALC_WORKING_DIR/execution/gazebo/setup_env.sh
export XPRA_SCALING=0
popd
echo "starting xpra"
xpra start --start="$ALC_HOME/utils/run.sh" --bind-tcp=0.0.0.0:10000 &
echo "finished starting xpra"
tail -f /dev/null
