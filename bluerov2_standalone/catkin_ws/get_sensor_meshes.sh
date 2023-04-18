FILE=/opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/dvl.dae
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    wget "https://raw.githubusercontent.com/uuvsimulator/uuv_simulator/master/uuv_sensor_plugins/uuv_sensor_ros_plugins/meshes/dvl.dae"
    sudo mkdir -p `dirname $FILE` 
    sudo mv dvl.dae /opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/dvl.dae
fi

FILE=/opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/pressure.dae
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    wget "https://raw.githubusercontent.com/uuvsimulator/uuv_simulator/master/uuv_sensor_plugins/uuv_sensor_ros_plugins/meshes/pressure.dae"
    sudo mkdir -p `dirname $FILE`
    sudo mv pressure.dae /opt/ros/melodic/share/uuv_sensor_ros_plugins/meshes/pressure.dae
fi
