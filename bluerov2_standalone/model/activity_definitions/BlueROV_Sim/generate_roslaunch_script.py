#!/usr/bin/env python
from __future__ import print_function
from future.utils import iteritems
from six import string_types

import os
import yaml
import argparse
import stat


parser = argparse.ArgumentParser(description='Script for running a simulation using the F1 Tenth Simulator setup.')
parser.add_argument('--output_dir', help='Directory for storing simulation logs and results')
parser.add_argument('--parameter_file', help='Simulation parameter file')
parser.add_argument('--ros_setup_file', help='ROS setup file for desired catkin workspace')
parser.add_argument('--launch_file', help='Default ROS package name and launch file', default=None)
parser.add_argument('--script_name', help='Directory for storing simulation logs and results', default='run_simulation.sh')
args = parser.parse_args()

# Load parameter file
print("Reading parameters from input file: %s" % args.parameter_file)
with open(args.parameter_file, 'r') as input_fp:
    parameters = yaml.load(input_fp)

# Check parameter file for any special cases. Update parameters accordingly
if parameters.get('random_seed', None) is None:
    if parameters.get('random_val'):
        parameters['random_seed'] = parameters['random_val']
    else:
        parameters['random_seed'] = '$RANDOM'

# Get launch file name & ROS Package
if parameters.get('launchfile', None):
    launchfile = parameters['launchfile']
    del parameters['launchfile']
else:
    launchfile = args.launch_file

#ros workspace
ros_workspace = '/alc_workspace/ros/devel/setup.bash'
if parameters.get('project_name', None):
    project_name = parameters['project_name']
    del parameters['project_name']
    ros_folder = os.path.join('/alc_workspace/ros', project_name, 'devel', 'setup.bash')
    if os.path.exists(ros_folder):
        ros_workspace = ros_folder

# Construct roslaunch command to execute
# Add --wait flag to wait for ROS core to be started at $ROS_MASTER_URI
print("Processing parameters file...")
#cmd  = "roslaunch %s --wait " %(launchfile)
cmd = ". %s --extend; roslaunch %s --wait " % (ros_workspace, launchfile)
for name, value in iteritems(parameters):
    # Adding parameters to the command line string
    cmd += name + ':='
    if isinstance(value, bool):
        cmd += str(int(value)) + ' '
    elif isinstance(value, string_types):
        # Wrap string types in quotes to avoid issues with whitespace
        cmd += '"' + str(value) + '" '
    else:
        cmd += str(value) + ' '
print("Done.")

# Set bag filename parameter
recording_filename = os.path.join(args.output_dir, 'recording.bag')
print("Setting bag file name to: %s" % recording_filename)
cmd = cmd + 'results_directory:=\"' + args.output_dir + '\" '
cmd = cmd + 'bag_filename:=\'\"' + recording_filename + '\"\' \n'

# Write launch command to script file
try:
    filename = os.path.join(args.output_dir, args.script_name)
    print('Writing script file = ' + filename)
    with open(filename, 'w') as script_file:
        # script_file.write('#!/usr/bin/env bash\n')
        script_file.write(cmd)
        # Set execute permission on output file
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)
    print('Script file created = ' + filename)
except Exception as e:
    print('Error while creating script file, message=' + str(e))
