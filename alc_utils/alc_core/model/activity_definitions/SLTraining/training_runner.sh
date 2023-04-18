#!/bin/bash
echo "Starting training runner script"

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
    exit -1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Output directory not set. Exiting."
    exit -1
fi

source /opt/ros/melodic/setup.bash

# Run verification script
echo "change directory to output directory"
cd  ${OUTPUT_DIR}

echo "update pythonpath"
export PYTHONPATH=$PYTHONPATH:$REPO_HOME:$REPO_HOME/alc_utils:$REPO_HOME/alc_utils/LaunchActivity:$REPO_HOME/alc_utils/ml_library_adapters:$ACTIVITY_HOME

echo "execute sl training"
python $ACTIVITY_HOME/SLTrainingRunner.py  "${INPUT_FILE}"

# Kill any child processes
echo "Killing child processes."
kill $(ps -s $$ -o pid=)

echo "Done."
exit 0
