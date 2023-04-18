#!/bin/sh
docker run --rm -it --runtime=nvidia -v "$MATLAB_ROOT":/usr/local/MATLAB/from-host -v "$MATLAB_ROOT":"$MATLAB_ROOT" -v $MATLAB_SUPPORT_ROOT:"$MATLAB_SUPPORT_ROOT" -v "$MATLAB_LOGS":/var/log/matlab -v $ALC_VERIVITAL_HOME:/verivital --mac-address="$MATLAB_MAC_ADDRESS" -e MATLAB_PATH="$MATLAB_ROOT" alc_jupyter_matlab sh -c  ". /setup.sh"

