- update setup_env.sh as per your setup
   - the MATLAB_SUPPORT_PACKAGE_ROOT can be obtained by typing the following command in the matlab shell  'matlabshared.supportpkg.getSupportPackageRoot'
- change the username 'matlabuser' in the Dockerfile based on the name of the user who is registed to run matlab
   - if all users on the machine can run matlab, please comment out the lines 11-13 and 18 in Dockerfile.

- run the following commands in order to build alc_jupyter_matlab
- . ./setup_env.sh
- ./build.sh
- ./run_setup.sh
  - This is interactive. One of the matlab dependencies used in verivital requires a user confirmation.

- At this stage you should have a complete alc_jupyter_matlab image

- restart services $ALC_HOME/docker/alc/run_services.sh 
