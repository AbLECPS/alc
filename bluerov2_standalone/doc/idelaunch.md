####BlueROV from command line

The instructions in this section describe how to run the BlueROV simulation natively on the host machine or using IDE available in the toolchain:
 - Go inside ROOT > ALC 
 - Launch `IDELanunch` plugin with `start` setting
 - select IDE selector, and click on VSCode
 - set up /etc/
 - when using IDE, use 'kinetic' for the commands

In the new browser (or on the host machine) tab open these 3 terminal windows:

Open three terminals:

**1.: Start roscore docker**
```
cd $ALC_HOME/catkin_ws
source run_roscore.sh
```
**2.: Start BlueROV docker**
```
cd $ALC_HOME/catkin_ws
source run_bluerov_sim.sh
```
In the docker:
```
source run_xvfb.sh
source src/vandy_bluerov/scripts/bluerov_launch.sh
```
**Preset scenarios**

If you want to run a preset scenario, run eg.: 
```
source src/vandy_bluerov/scripts/cp1_00.sh
```
Check activity definitions for more detail about available scenarios from command line and toolchain.

**3.: Optional for visual representation: start rviz**
```
cd $ALC_HOME/catkin_ws
source ros_<kinetic/melodic>_setup_gazebo_ros_connection.sh
source $ALC_HOME/bluerov2/catkin_ws/src/vandy_bluerov/scripts/run_rviz_bluerov2.sh 
```
If you are using the web IDE, inside ROOT > ALC select IDE selector, and click on VNC. A new tab will open in the browser with the RViz running inside


**4.: Optional for visual representation: start RQT PyTrees BT graph**


***Additional installation for host machine***

Install RQT PyTrees viewer on host machine:
```
sudo apt install ros-<kinetic/melodic>-rqt-py-trees
```

To use RQT run:

```
cd $ALC_HOME/catkin_ws
source ros_<kinetic/melodic>_setup_gazebo_ros_connection.sh
rqt_py_trees
```

If you are using the web IDE, inside ROOT > ALC select IDE selector, and click on VNC. A new tab will open in the browser with the RQT running inside.
Select `/BlueROV_tree/log/tree` from the BTree list


**Known issues**
-  Running simulation in IDE or on the host machine, Gazebo sometimes dies at the beginning of the simulation like this:

```
[gazebo-35] process has died [pid 5144, exit code 134, cmd /opt/ros/kinetic/lib/gazebo_ros/gzserver -u --verbose -e ode --seed 27168 worlds/environment_empty.world __name:=gazebo __log:=/root/.ros/log/163f34f0-1a0a-11ec-b167-0242ac120002/gazebo-35.log].
log file: /root/.ros/log/163f34f0-1a0a-11ec-b167-0242ac120002/gazebo-35*.log
```
In this case you have to terminate and restart simulation.


-  After launching RViz, ignore these errors:
```
[ERROR] [1632144553.764106835]: Robot model parameter not found! Did you remap 'robot_description'?
```

-  Running RViz in IDE > VNC window has some limitations caused by the virtual GUI. Eg. if you can't see the bluer pipeline in the beginning of the simulation, you can reopen the .rviz file from `File` menu.
-  The UUV green path cannot be seen also in this way.
-  After each simulation, you have to reopen RViz to visualize the running experiment.
