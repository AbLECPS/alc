#!/bin/bash
set -e

export PATH=$MATLAB_PATH/bin:$PATH
matlab_setup_dir="$MATLAB_PATH/extern/engines/python"
if [ -d "${matlab_setup_dir}" ]; then
  pushd $MATLAB_PATH/extern/engines/python
  sudo python setup.py install
  popd
else
  echo "WARNING: Required MATLAB directory (${matlab_setup_dir}) does not exist within Jupyter-Matlab docker image. Continuing without MATLAB."
fi


echo "current dir"
echo pwd

echo "JUPYTER WORK DIR"
echo $JUPYTER_WORK_DIR
cd $JUPYTER_WORK_DIR
#ls /verivital
#cp /verivital/pathdef.m .
echo "current directory" 
pwd
echo "running jupyter command"
jupyter notebook --allow-root --config=/jupyter_notebook_config.py
