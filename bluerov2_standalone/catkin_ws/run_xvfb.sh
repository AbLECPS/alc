#!/bin/bash

. devel/setup.bash
echo "$(tput setaf 1)$(tput setab 7)VU BlueROV2 codebase $(tput sgr 0)"
echo "To run BlueROV2 simulation use:"
echo "source src/vandy_bluerov/scripts/{selected_launchfile}.sh"
echo "eg.:"
echo "source src/vandy_bluerov/scripts/bluerov_launch.sh"

#
source /tf2_ros/devel/setup.bash 
source /aa/devel/setup.bash --extend
source /opt/ros/melodic/setup.bash --extend
source /cvbridge_build_ws/devel/setup.bash --extend
Xvfb :1 -screen 0 800x600x16 &
export DISPLAY=:1.0

#socat -d -d pty,raw,echo=0 pty,raw,echo=0 &
export PYTHONPATH=$PYTHONPATH:$ALC_HOME/alc_utils/assurance_monitor:$ALC_HOME/alc_utils:$ALC_HOME:/aa/src/vandy_bluerov/nodes/

