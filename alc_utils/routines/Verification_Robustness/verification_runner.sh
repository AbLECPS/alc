#!/bin/bash
echo "Starting verification runner script"

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

# Installing python matlab engine
echo "Install python matlab engine"
cd /usr/local/MATLAB/from-host/extern/engines/python
python3.6 setup.py install

#Installing imageio
echo "Install imageio"
pip install imageio

# Run verification script
echo "change directory to output directory" ${}
cd  ${OUTPUT_DIR}

echo "update pythonpath"
export PYTHONPATH=$PYTHONPATH:$ALC_HOME/webgme/ALC-Dev/src/plugins/LaunchVerification:$ALC_HOME:$ALC_HOME/alc_utils:$ALC_HOME/alc_utils/LaunchActivity

echo "execute robustness verification"
python3.6 $ALC_HOME/webgme/ALC-Dev/src/plugins/LaunchVerification/VerificationRobustnessRunner.py  "${INPUT_FILE}"


echo "Done."
exit 0
