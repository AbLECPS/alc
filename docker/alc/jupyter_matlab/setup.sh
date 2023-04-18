export PATH=$MATLAB_PATH/bin:$PATH
matlab -nodisplay -nosplash -nodesktop -r "run('/setup_verivital.m');quit;"
cd /usr/local/MATLAB/from-host/extern/engines/python
sudo python setup.py install

