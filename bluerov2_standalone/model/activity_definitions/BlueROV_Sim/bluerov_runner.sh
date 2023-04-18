#!/bin/bash
echo "Starting bluerov_runner script"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

# Get arguments
case $key in
    -i|--input_file)
    INPUT_FILE="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--output_dir)
    OUTPUT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo INPUT FILE         = "${INPUT_FILE}"
echo OUTPUT DIRECTORY   = "${OUTPUT_DIR}"

#if [[ -n $1 ]]; then
#    echo "Last line of file specified as non-opt/last argument:"
#    tail -1 "$1"
#fi

# Check that required arguments are set
if [[ -z "$INPUT_FILE" ]]; then
    echo "Input file not set. Exiting."
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Output directory not set. Exiting."
    exit 1
fi

echo "USER:PWD"
echo $USER:$PWD

echo "ALC_HOME"
echo $ALC_HOME

echo "REPO_HOME"
echo $REPO_HOME

echo "PYTHONPATH"
echo $PYTHONPATH

echo "ALC_WORKING_DIR"
echo ${ALC_WORKING_DIR}

# Make symbolic python link where ROS expects to find python
ln -s /usr/bin/python /usr/local/bin/python

# Source catkin setup files
echo "Sourcing catkin setup file"
echo $PWD
. /aa/devel/setup.bash
. /tf2_ros/devel/setup.bash --extend
. /opt/ros/melodic/setup.bash --extend
. /cvbridge_build_ws/devel/setup.bash --extend


# Setup virtual frame buffer
Xvfb :1 -screen 0 800x600x16 &
export DISPLAY=:1.0

export PYTHONPATH=$PYTHONPATH:$REPO_HOME/alc_utils:$PYTHONPATH:$REPO_HOME/alc_utils/config:$REPO_HOME/alc_utils/assurance_monitor:$REPO_HOME/alc_utils/ml_library_adapters:$REPO_HOME/alc_utils:${ACTIVITY_HOME}
export PYTHONPATH=$PYTHONPATH:$ALC_HOME:$ALC_WORKING_DIR
export PYTHONPATH=$PYTHONPATH:/aa/src/vandy_bluerov/nodes/



# Use python to parse parameter file and generate a bash script for running simulation
echo "Creating follow LEC roslaunch script with python generate_roslaunch_script"
echo "python version"
python --version

echo "python3 version"
python3 --version

python ${ACTIVITY_HOME}/generate_roslaunch_script.py \
    --parameter_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ros_setup_file "/aa/devel/setup.bash" \
    --launch_file "vandy_bluerov start_bluerov_simulation.launch"


# Run generated launch script
echo "Running simulation launch script"
export
. "${OUTPUT_DIR}/run_simulation.sh"

# Kill any child processes
echo "Killing child processes."
kill $(ps -s $$ -o pid=)
echo "Done."
exit 0
