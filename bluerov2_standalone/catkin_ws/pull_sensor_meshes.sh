FILE=$PWD/src/bluerov2/bluerov2_description/meshes/dvl.dae
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    wget "https://raw.githubusercontent.com/uuvsimulator/uuv_simulator/master/uuv_sensor_plugins/uuv_sensor_ros_plugins/meshes/dvl.dae"
    mkdir -p `dirname $FILE` 
    mv dvl.dae $FILE
fi

FILE=$PWD/src/bluerov2/bluerov2_description/meshes/pressure.dae
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    wget "https://raw.githubusercontent.com/uuvsimulator/uuv_simulator/master/uuv_sensor_plugins/uuv_sensor_ros_plugins/meshes/pressure.dae"
    mkdir -p `dirname $FILE`
    mv pressure.dae $FILE    
fi
