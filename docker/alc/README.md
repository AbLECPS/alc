**WARNING:**
The information in this file is long outdated as of Aug 17, 2021.
Some points may still be correct, but many details are likely wrong.
Highly recommend the reader consult the full ALC Documentation for more up-to-date information.


Contents
--------
This folder contains two primary scripts for building and launching ALC-Toolchain related docker images:

  1) **setup.sh** - Performs initial setup steps (see file for details) and builds various base docker images 
  (including simulation environment images).
  
  2) **run_services.sh** - Checks the environment configuration, then starts all docker services in the 
  docker-compose.yml file.

**See ALC Documentation for full details on starting/building the ALC Toolchain**

Assumptions / Updates to be made to build script before execution.
------------------------------------------------------------------

1. The setup script assumes that sources are available in the following folders:
    - The alc sources are in $ALC_HOME
    - The alc webgme sources are in $ALC_HOME/webgme

2. The build script assumes that the following folders can be created and mounted to some of the docker containers.
    - export ALC_WORKSPACE=/home/alc/alc_workspace/
    - export ALC_JUPYTER_WORKDIR=/home/alc/alc_workspace/jupyter
    - export ALC_FILESERVER_ROOT=/home/alc/alc_fileshare_workspace/
    - export ALC_DOCKERFILES=/home/alc/alc_dockerfiles/


Assumptions regarding docker images
-----------------------------------
Several of the ALC docker images are built from the **build_tensorflow** image. 
By default, this image uses the standard tensorflow-gpu docker distribution image (version 1.13.0 at time of writing). 
However, for users who need to build tensorflow from source, the Dockerfile for this image can be updated. 


Setup
-----------
Define required environmental variables, then run 

    ./setup.sh --build

Start the docker services
-------------------------
Run the command:

    ./run_services.sh --build

The WebGME service should then be available at http://localhost:8000 
(or at user-defined port if specified).






