#!/bin/bash

# Exit on error, don't suppress
set -e

source /opt/ros/melodic/setup.bash

# Execute job
python main.py
