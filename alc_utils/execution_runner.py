#!/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines the ExecutionRunner utility class for executing experiments in a docker environment."""

import json
import os
import time
from alc_utils import common as alc_common
from alc_utils import config as alc_config
import random
import docker
import docker.errors
import requests.exceptions
import logging
import sys
import rospy
import rosgraph
import roslib.message
from std_msgs.msg import Float32
import socket
import errno
import signal
from functools import partial
from future.utils import viewitems


DOCKER_ERROR_CONFLICT_CODE = 409
# How long to wait between attempts to contact ROS master (in seconds)
ROS_MASTER_CONTACT_DELAY_S = 2
# How many attempts to contact the ROS master should be made
ROS_MASTER_CONTACT_ATTEMPTS = 4
DOCKER_WAIT_TIMEOUT_S = 1
DOCKER_STOP_TIMEOUT_S = 10
DOCKER_TIMEOUT_BUFFER_S = 60
LOGGING_LEVEL = logging.DEBUG
repo_home_key = 'REPO_HOME'


class ExecutionRunner:
    """This class configures a docker environment and executes an experiment based on the provided execution file."""

    def __init__(self, execution_file, timeout_buffer=DOCKER_TIMEOUT_BUFFER_S):
        """Creates an instance of the ExecutionRunner class with parameters loaded from the specified execution file."""
        # Configuration parameters and default values
        self.execution_file = execution_file
        self._timeout_buffer = timeout_buffer

        # Source ALC environment variables
        alc_common.source_env_vars(alc_config.env)

        # Open and load execution file
        with open(self.execution_file, 'r') as execution_fp:
            self._execution_dict = json.load(execution_fp)
        self._execution_name = self._execution_dict['name']
        base_dir = os.path.expandvars(self._execution_dict['base_dir'])
        if os.path.isabs(base_dir):
            self._base_dir = base_dir
        else:
            self._base_dir = os.path.join(
                alc_config.WORKING_DIRECTORY, base_dir)
        self._results_dir = os.path.join(
            self._base_dir, self._execution_dict['results_dir'])
        self._timeout = self._execution_dict['timeout']
        self._unique_containers = self._execution_dict.get(
            "unique_containers", True)
        self._echo_docker_logs = self._execution_dict.get("echo_logs", True)
        self._activity_home = self._execution_dict.get('activity_home', '')
        if (self._activity_home):
            self._activity_home = os.path.expandvars(self._activity_home)
            if os.path.isabs(self._activity_home):
                self._activity_home = self._activity_home
            else:
                self._activity_home = os.path.join(
                    alc_config.WORKING_DIRECTORY, self._activity_home)
            print('activity_home = '+self._activity_home)

        # Ensure results directory exists
        if not os.path.isdir(self._results_dir):
            os.makedirs(self._results_dir)

        # Variable init
        self.docker_client = docker.from_env()
        self.docker_api_client = docker.APIClient(
            base_url='unix://var/run/docker.sock')
        self.docker_network = None
        self.self_docker_container = None
        self.unique_suffix = None
        self.child_containers = []
        self.ros_master_container = None
        self.ros_master_ip = None
        self.netmask_ip = None
        self.ros_listener_node = None

        # Setup logger
        self._logger = logging.getLogger(
            'execution_runner_%s' % self._execution_name)
        if len(self._logger.handlers) == 0:  # FIXME: What is this check for?
            # Log to stdout
            self._out_hdlr = logging.StreamHandler(sys.stdout)
            self._out_hdlr.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
            self._out_hdlr.setLevel(LOGGING_LEVEL)
            self._logger.addHandler(self._out_hdlr)
            self._logger.setLevel(LOGGING_LEVEL)

            # Log to file
            log_filename = os.path.join(
                self._results_dir, 'execution_runner.log')
            self._file_hdlr = logging.FileHandler(log_filename)
            self._file_hdlr.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
            self._file_hdlr.setLevel(LOGGING_LEVEL)
            self._logger.addHandler(self._file_hdlr)
            self._logger.setLevel(LOGGING_LEVEL)

        # Register signal handler function
        signal.signal(signal.SIGINT, self.sighandler)
        signal.signal(signal.SIGTERM, self.sighandler)

    def sighandler(self, signum, frame):
        try:
            self._logger.debug(
                "ExecutionRunner sighandler invoked by signal number %d." % signum)
            self.cleanup(fetch_output=False)
        except:
            print('exception in sighandler during logging/ cleanup')

    def __del__(self):
        try:
            if (self._logger):
                self._logger.debug("ExecutionRunner delete function invoked.")
            self.cleanup(fetch_output=False)
        except:
            print('exception in _del_ during logging/ cleanup')

    # Function to create unique docker network by appending random integer.
    # If network already exists, will attempt to generate a new network name multiple times.
    # Throws RuntimeError if no unique network found after specified number of attempts
    def create_unique_docker_network(self):
        attempt_count = 0
        while attempt_count < 5:
            attempt_count += 1

            # Try to create network with random name.
            rand_int = random.randint(0, 1000)
            network_name = "network_%d" % rand_int
            try:
                network = self.docker_client.networks.create(
                    name=network_name, driver="bridge", check_duplicate=True)
            except docker.errors.APIError as e:
                # Check if network already exists
                if e.status_code == DOCKER_ERROR_CONFLICT_CODE:
                    continue
                else:
                    raise e

            # Return network if successful
            self._logger.info(
                "Created docker network with name %s" % network.name)
            return network, rand_int

        # Failed after multiple attempts
        raise RuntimeError("ExecutionRunner failed to create a unique network for docker containers after %d attempts"
                           % attempt_count)

    def run(self):
        """Execute the experiment as specified in the execution file.

        Configures and spawns a docker environment with the appropriate container images, then executes commands
        specified in the execution deployment file."""
        self._logger.info('Running execution: ' + self._execution_name)
        self._logger.info('Base directory: ' + self._base_dir)
        self._logger.info('Results directory: ' + self._results_dir)

        # Start docker containers
        self.run_containers()

        # Get docker container handle to container running this script (or None if running natively)
        self.self_docker_container = None
        self_name = os.getenv("HOSTNAME", None)
        if self_name is None:
            self._logger.warning(
                "Environment variable 'HOSTNAME' undefined - assuming worker is executing on host.")
        else:
            self._logger.info(
                "Searching for docker container '%s' hosting the ExecutionRunner process..." % self_name)
            try:
                self.self_docker_container = self.docker_client.containers.get(
                    self_name)
                self._logger.info(
                    "Found docker container '%s' hosting the ExecutionRunner process.")
            except docker.errors.NotFound as e:
                self._logger.warning("Docker could not find container with name: %s. "
                                     "This is expected if ExecutionRunner is running natively" % self_name)
                if e.message is not None:
                    self._logger.warning("Docker error: ", str(e.message))

        # If worker is executing inside docker container, add to newly created docker network.
        # Find local IP of this container, or of the host machine if running natively
        if self.self_docker_container is not None:
            self._logger.info("Adding host container (%s) to docker network (%s)" % (
                self.self_docker_container.name, self.docker_network.name))
            self.docker_network.connect(self.self_docker_container)
            container_info = self.docker_api_client.inspect_container(
                self.self_docker_container.id)
            container_ip = container_info["NetworkSettings"]["Networks"][self.docker_network.name]["IPAddress"]
        else:
            self._logger.info(
                "No host docker container found - assuming ExecutionRunner is executing natively.")
            network_info = self.docker_api_client.inspect_network(
                self.docker_network.id)
            container_ip = network_info["IPAM"]["Config"][0]["Gateway"]

        # FIXME: Don't like having yet another time-delay. Can this be done better?
        #        If not, at least make it configurable
        # Wait for nodes to initialize
        time.sleep(15)

        # Start ROS node to listen for termination signal
        # Note: This can be started only if there is a ros-master
        if (self.ros_master_container):
            listener_kwargs = {"ros_master_uri": "http://%s:11311" % self.ros_master_ip, "ros_ip": container_ip,
                               "ros_log_dir": self._results_dir, "logger": self._logger,
                               "termination_topic": self._execution_dict.get("termination_topic", "/alc/stopsim"),
                               "echo_topics": self._execution_dict.get("echo_topics", None)}
            self.ros_listener_node = start_ros_listener(**listener_kwargs)

        # Simple polling loop to check if all containers have exited or if timeout has been reached.
        # Timeout is a security measure in case something happens, e.g. roscore not responding
        if self._timeout is None:
            stop_time = None
            self._logger.warn(
                'No timeout configured. Will run until docker images shutdown.')
        else:
            stop_time = time.time() + self._timeout + self._timeout_buffer
            self._logger.info('Timeout is set to t=%.f s' %
                              (self._timeout + self._timeout_buffer))

        while True:
            # # Check if all child containers have finished
            # all_containers_exited = True
            # for child in self.child_containers:
            #     child.reload()
            #     if child.status != 'exited':
            #         all_containers_exited = False
            #         break

            # Check if any child container has finished
            containers_finished = False
            for child in self.child_containers:
                child.reload()
                if child.status != 'running':
                    containers_finished = True
                    break
            if containers_finished:
                self._logger.info(
                    'At least one container has exited. Assuming experiment complete.')
                break

            # Stop any running containers if termination signal has been received
            if (self.ros_listener_node is not None) and (self.ros_listener_node.read_termination_flag()):
                self._logger.info(
                    "Execution runner received termination signal from ROS node. Stopping containers...")
                break

            # TODO: If timeout is reached before containers shutdown, data may be invalid/corrupted.
            #       Should report error back to WebGME user.
            # TODO: Check if "recording.bag.active" file exists. Wait until it is removed, or at least warn user
            # Exit loop and let cleanup function stop any running containers if timeout has been reached
            if (stop_time is not None) and (time.time() > stop_time):
                self._logger.warning(
                    'Execution timeout reached before any docker container exited. Stopping containers...')
                break

            # Delay before next iteration
            time.sleep(1)

        # Cleanup docker containers and network
        try:
            self.cleanup(fetch_output=self._echo_docker_logs)
        except:
            print('exception in cleanup during run')

        # Clear docker variables
        self.docker_network = None
        self.unique_suffix = None
        self.child_containers = []
        self.ros_master_container = None

        return 0, self._base_dir

    def run_containers(self):
        # Find all necessary docker image names from execution file
        # If we are running a ROS-based setup, will need a ros_master container. Check if this is set
        ros_master_image = self._execution_dict["ros_master_image"]
        if ros_master_image is not None:
            required_docker_image_names = [ros_master_image]
        else:
            required_docker_image_names = []
        for container_dict in self._execution_dict["containers"]:
            required_docker_image_names.append(container_dict["image"])

        # Ensure all necessary docker images have already been built
        available_docker_image_names = []
        for image in self.docker_client.images.list():
            available_docker_image_names.extend(image.tags)
        for image_name in required_docker_image_names:
            assert image_name in available_docker_image_names, "Docker image '%s' not found." % image_name

        # Create docker network
        self.docker_network, integer_suffix = self.create_unique_docker_network()
        self.unique_suffix = str(integer_suffix)

        # Find network IP range strip digits after last '.' from IP
        network_info = self.docker_api_client.inspect_network(
            self.docker_network.id)
        gateway_ip = network_info['IPAM']['Config'][0]['Gateway']
        self.netmask_ip = ".".join(gateway_ip.split('.')[0:-1]) + '.'
        self._logger.info("Docker network IP range: %s<X>" % self.netmask_ip)

        # FIXME: Is it possible to mount source tree in read-only mode?
        # Host volumes to mount on all docker containers
        repo_home = os.environ.get(repo_home_key, '')
        os.environ['ALC_HOME'] = alc_config.ALC_HOME
        if (repo_home):
            os.environ['ALC_HOME'] = os.environ[repo_home_key]
        source_tree_dir = alc_config.ALC_HOME
        workspace_dir = alc_config.WORKING_DIRECTORY
        docker_volumes = {source_tree_dir: {'bind': source_tree_dir, 'mode': 'rw'},
                          workspace_dir: {'bind': workspace_dir, 'mode': 'rw'}}

        # Ensure ROS logging directory exists
        ros_log_dir = os.path.join(self._results_dir, 'ros_logs')
        if not os.path.isdir(ros_log_dir):
            os.makedirs(ros_log_dir)

        # Set default environment variables for docker containers
        # FIXME: Currently rely on docker API to assign IPs in a consistent order
        #   It would be better to assign them manually, but this is more difficult than it sounds
        if self._unique_containers:
            ros_master_name = "ros-master_%s" % self.unique_suffix
        else:
            ros_master_name = "ros-master"
        self.ros_master_ip = self.netmask_ip + "2"
        docker_env_vars = alc_config.env.copy()
        docker_env_vars["ALC_HOME"] = os.environ['ALC_HOME']
        if (repo_home):
            docker_env_vars[repo_home_key] = os.environ[repo_home_key]
        docker_env_vars["ALC_WORKING_DIR"] = alc_config.WORKING_DIRECTORY
        docker_env_vars["ROS_MASTER_URI"] = "http://%s:11311" % ros_master_name
        docker_env_vars["ROS_LOG_DIR"] = ros_log_dir
        docker_env_vars["PYTHONPATH"] = os.environ['ALC_HOME']
        docker_env_vars["ROS_IP"] = self.ros_master_ip
        if (self._activity_home):
            docker_env_vars["ACTIVITY_HOME"] = self._activity_home

        # Start ROS Core container if desired
        if ros_master_image is not None:
            self._logger.info('Starting ROS Core docker container.')
            self.ros_master_container = self.docker_client.containers.run(ros_master_image, 'roscore',
                                                                          detach=True, hostname='ros-master',
                                                                          name=ros_master_name,
                                                                          volumes=docker_volumes,
                                                                          environment=docker_env_vars,
                                                                          network=self.docker_network.name)
            self._logger.info('ROS Core running at https://%s:11311 on docker network %s.'
                              % (ros_master_name, self.docker_network.name))
        else:
            self.ros_master_container = None

        # Set default docker run arguments
        default_kwargs = {"detach": True,
                          "network": self.docker_network.name,
                          "working_dir": self._base_dir}

        # Spawn new container for each docker
        # TODO: Log docker output to file in realtime, not after completion
        for container_dict in self._execution_dict["containers"]:
            # Determine unique IP address for this container
            # First container IP starts at <netmask_ip>.3 and goes up from there (eg. 192.168.0.<3,4,5...>)
            container_ip = self.netmask_ip + \
                str(len(self.child_containers) + 3)

            # Add container-specific keyword args to default options
            container_kwargs = default_kwargs.copy()
            name = container_dict["name"]
            if self._unique_containers:
                unique_name = name + '_%s' % self.unique_suffix
            else:
                unique_name = name
            container_env_vars = docker_env_vars.copy()
            container_env_vars["ROS_IP"] = container_ip
            container_kwargs["name"] = unique_name
            container_kwargs["hostname"] = name
            container_kwargs["environment"] = container_env_vars
            container_kwargs["volumes"] = docker_volumes.copy()

            # Add/override keyword args with additional options specified in execution file
            for key, value in viewitems(container_dict["options"]):

                # For mount volumes or environment variables, add to existing defaults
                # These options are specified as dictionaries, so "value" here is another dictionary
                if key == "volumes":
                    updated_arg = container_kwargs[key]
                    for updated_key, updated_value in viewitems(value):
                        # Host directory to mount may be specified with environment variables. Expand before adding.
                        expanded_path = os.path.expandvars(updated_key)
                        expanded_values = updated_value
                        expanded_bind_value = os.path.expandvars(
                            updated_value["bind"])
                        expanded_values["bind"] = expanded_bind_value
                        updated_arg[expanded_path] = expanded_values
                    container_kwargs[key] = updated_arg
                elif key == "environment":
                    updated_arg = container_kwargs[key]
                    for updated_key, updated_value in viewitems(value):
                        # Docker environment variables may be specified in terms of host env vars. Expand before adding.
                        expanded_path = os.path.expandvars(updated_value)
                        updated_arg[updated_key] = expanded_path
                    container_kwargs[key] = updated_arg

                # Otherwise, add/override option with new value
                else:
                    container_kwargs[key] = value

            # Append any required arguments to container command
            full_input_file = os.path.join(
                self._base_dir, container_dict["input_file"])
            container_command = ""
            self._logger.info(str(container_kwargs.keys()))
            if ("entrypoint" not in container_kwargs):
                container_command += "sh -c '"
            container_command += "echo $ALC_HOME; echo $ALC_WORKING_DIR; echo $ACTIVITY_HOME;"
            container_command += os.path.expandvars(container_dict["command"]) + " --input_file \"%s\" --output_dir \"%s\"" \
                % (full_input_file, self._results_dir) + " exit 0"
            if ("entrypoint" not in container_kwargs):
                container_command += "'"

            # Start Docker container
            self._logger.info('Starting %s docker container with command "%s".' % (
                name, container_command))
            container = self.docker_client.containers.run(container_dict["image"], container_command,
                                                          **container_kwargs)
            self._logger.info('Container started.')

            # Add new container to child_containers
            self.child_containers.append(container)

        if self._execution_dict.get("gazebo_vis_script", False):
            self.write_execution_script()

    def write_execution_script(self):
        """Write a script for launching the Gazebo client and connecting to the correct ROS master.
        Used for visualization of simulation."""
        self._logger.info('Generating gazebo client execution file...\n')

        # Ensure directory exists
        script_dir = os.path.join(
            alc_config.WORKING_DIRECTORY, 'execution', 'gazebo')
        alc_common.mkdir_p(script_dir)

        # Content of gazebo client execution script
        exec_cmd = "#!/bin/bash\n"
        exec_cmd += 'export ROS_MASTER_URI=http://%s:11311\n' % self.ros_master_ip
        exec_cmd += 'export GAZEBO_MASTER_URI=http://%s4:11345\n' % self.netmask_ip

        # Open file and write execution script
        exec_filename = os.path.join(script_dir, 'setup_env.sh')
        f = open(exec_filename, "w")
        f.write(exec_cmd)
        f.close()

        exec_cmd += 'gzclient --verbose'

        # Open file and write execution script
        exec_filename = os.path.join(script_dir, 'rungzclient.sh')
        f = open(exec_filename, "w")
        f.write(exec_cmd)
        f.close()

        self._logger.info(
            'Wrote gazebo client execution file to %s.\n' % exec_filename)

        # Ensure directory exists
        script_dir = os.path.join(
            alc_config.WORKING_DIRECTORY, 'execution', 'plotter')
        alc_common.mkdir_p(script_dir)

        # Content of gazebo client execution script
        exec_cmd = "#!/bin/bash\n"
        exec_cmd += 'export ROS_MASTER_URI=http://%s:11311\n' % self.ros_master_ip
        exec_cmd += 'rosrun  plotjuggler PlotJuggler'

        # Open file and write execution script
        exec_filename = os.path.join(script_dir, 'runplotjuggler.sh')
        f = open(exec_filename, "w")
        f.write(exec_cmd)
        f.close()

        self._logger.info(
            'Wrote plotter execution file to %s.\n' % exec_filename)

    def cleanup(self, fetch_output=False):
        # Want to raise exception if error is encountered, but need to finish cleanup first.
        error_encountered = False

        # Apparently this function is also blocking with no option for a timeout
        rospy.signal_shutdown("ExecutionRunner cleanup function called.")

        # Make sure docker children are still valid and up-to-date objects. Remove them if not
        children = []
        for child in self.child_containers:
            try:
                child.reload()
            except docker.errors.NotFound:
                continue
            children.append(child)

        # Stop each child container if they have not already exited
        for child in children:
            if child.status == 'exited':
                if (self._logger):
                    self._logger.info(
                        'Docker container "%s" has already exited.' % child.name)
            else:
                if (self._logger):
                    self._logger.info(
                        'Docker container "%s" in state "%s". Sending stop command...' % (child.name, child.status))
                try:
                    child.stop(timeout=DOCKER_STOP_TIMEOUT_S)
                    if (self._logger):
                        self._logger.info('Container stopped successfully.')
                except docker.errors.APIError as e:
                    error_encountered = True
                    if (self._logger):
                        self._logger.error('Docker APIError occured with status code %d when stopping container %s'
                                           % (e.status_code, child.id))
                        self._logger.error('APIError message: %s' % e.message)

        # ROS Master runs indefinitely, so docker container never shuts down.
        # Stop it here once all child containers have either exited cleanly or been stopped
        if self.ros_master_container is not None:
            if (self._logger):
                self._logger.info('Stopping ROS master container...')
            try:
                
                self.ros_master_container.stop(timeout=DOCKER_STOP_TIMEOUT_S)
                if (self._logger):
                    self._logger.info('Container stopped successfully.')
            except docker.errors.APIError as e:
                error_encountered = True
                if (self._logger):
                    self._logger.error('Docker APIError occured with status code %d when stopping container %s'
                                       % (e.status_code, self.ros_master_container.id))
                    self._logger.error('APIError message: %s' % e.message)

            # ROS Master can now be added to children for convenience
            children.append(self.ros_master_container)

        container_output_strs = {}
        if fetch_output:
            # Fetch and save STDOUT/STDERR output from each docker container to file
            # Also store output messages to be printed at end of execution (Echoing output here makes debugging easier)
            for child in children:
                try:
                    # Fetch output and store
                    output_str = child.attach(
                        stdout=True, stderr=True, logs=True)
                    container_output_strs[child.name] = output_str.decode('utf-8').strip()

                    # Write output to log file
                    log_file_name = os.path.join(
                        self._results_dir, 'container_%s.log' % child.name)
                    log_fp = open(log_file_name, 'wb')
                    log_fp.write(output_str)
                    if (self._logger):
                        self._logger.info("Wrote container '%s' output to log file: %s" % (
                            child.name, log_file_name))
                except docker.errors.APIError as e:
                    error_encountered = True
                    if (self._logger):
                        self._logger.error('Docker APIError occured with status code %d when reading logs from container %s'
                                           % (e.status_code, child.id))
                        self._logger.error('APIError message: %s' % e.message)

        # Ensure containers have shutdown and fetch their exit codes, then remove containers
        for child in children:
            try:
                if (self._logger):
                    self._logger.info("Waiting on container '%s'..." % child.name)
                results = child.wait(timeout=DOCKER_WAIT_TIMEOUT_S)
                if results["StatusCode"] != 0:
                    if (self._logger):
                        self._logger.error("Docker container %s returned non-zero exit code %str." % (
                            child.id, str(results["StatusCode"])))
                    error_encountered = True
            except requests.exceptions.ReadTimeout as e:
                error_encountered = True
                if (self._logger):
                    self._logger.error(
                        'Timeout while waiting on container %s to exit.' % child.id)
                    self._logger.error('APIError message: %s' % e.message)
            except docker.errors.APIError as e:
                error_encountered = True
                if (self._logger):
                    self._logger.error('Docker APIError occured with status code %d when waiting on container %s'
                                       % (e.status_code, child.id))
                    self._logger.error('APIError message: %s' % e.message)

            try:
                if (self._logger):
                    self._logger.info("Removing container '%s'..." % child.name)
                child.remove()
                if (self._logger):
                    self._logger.info('Container removed successfully.')
            except docker.errors.APIError as e:
                error_encountered = True
                if (self._logger):
                    self._logger.error('Docker APIError occured with status code %d when removing container %s'
                                       % (e.status_code, child.id))
                    self._logger.error('APIError message: %s' % e.message)

        # Cleanup generated network
        if self.docker_network is not None:
            try:
                # Remove self from docker network if running in a docker container
                if self.self_docker_container is not None:
                    self.docker_network.disconnect(
                        self.self_docker_container, force=True)

                # Remove docker network
                if (self._logger):
                    self._logger.info('Removing docker network "%s"...' %
                                      self.docker_network.name)
                self.docker_network.remove()
                if (self._logger):
                    self._logger.info('Network removed successfully.')
            except docker.errors.APIError as e:
                error_encountered = True
                if (self._logger):
                    self._logger.error('Docker APIError occured with status code %d when removing network %s'
                                       % (e.status_code, self.docker_network.id))
                if (self._logger):
                    self._logger.error('APIError message: %s' % e.message)

        if fetch_output:
            # Echo output from containers here for easier debugging
            for key, value in viewitems(container_output_strs):
                if (self._logger):
                    self._logger.info(
                        "****************** CONTAINER %s OUTPUT ******************" % key)
                    self._logger.info(value)
                    self._logger.info(
                        "**************** END CONTAINER %s OUTPUT ****************" % key)

        # Cleanup variables pointing to now defunct containers
        self.ros_master_container = None
        self.child_containers = []

        # Now that cleanup is complete, raise an exception if any errors were encountered
        if error_encountered:
            # raise RuntimeError("Encountered error while performing cleanup. See log for additional details.")
            if (self._logger):
                self._logger.error(
                    "Encountered error while performing cleanup. See log for additional details.")


# Function to start ROS node which listens for termination signal.
# Python logger claims it is thread-safe, so using ExecutionRunner logger here should be OK.
def start_ros_listener(ros_master_uri=None, ros_ip=None, ros_log_dir=None,
                       logger=None, termination_topic="/alc/stopsim", echo_topics=None):
    attempts = 0

    os.environ["ROS_MASTER_URI"] = ros_master_uri
    os.environ["ROS_IP"] = ros_ip
    os.environ["ROS_HOSTNAME"] = ros_ip
    os.environ["ROS_LOG_DIR"] = ros_log_dir

    logger.info("Initializing ROS listener node with master URI: %s" %
                os.environ["ROS_MASTER_URI"])
    logger.info("Listener using ROS_IP: %s" % os.environ["ROS_IP"])

    # Check that we can connect to the master
    # This is done to prevent rospy.init_node() from blocking indefinitely if master is unreachable
    found_master = False
    while attempts < ROS_MASTER_CONTACT_ATTEMPTS:
        attempts += 1
        try:
            # This method of checking for master is taken from rostopic._check_master() function.
            # Not well documented, but exists in ROS source code
            rosgraph.Master('/rostopic').getPid()
            found_master = True
        except socket.error as e:
            #
            # if e.errno != errno.ECONNREFUSED:
            #    raise e
            # else:
            logger.info(
                "Unable to contact ROS master on attempt %d." % attempts)
            if attempts == ROS_MASTER_CONTACT_ATTEMPTS:
                break
        time.sleep(ROS_MASTER_CONTACT_DELAY_S)

    if found_master:
        logger.info("Connection to ROS master successful.")
    else:
        logger.warning(
            "Failed to connect to ROS master. Skipping ROS listener initialization.")
        return None

    logger.info("Initializing ROS Listener node...")
    rospy.init_node('worker_ros_listener',
                    log_level=rospy.INFO, disable_signals=True)
    ros_listener_node = Listener(
        logger, termination_topic=termination_topic, echo_topics=echo_topics)
    logger.info("ROS node initalized.")
    return ros_listener_node


class Listener:
    def __init__(self, logger, termination_topic=None, echo_topics=None):
        self._logger = logger

        # Subscribe to termination topic and init variables
        if termination_topic is not None:
            self._termination_topic = termination_topic
            self._termination_sub = rospy.Subscriber(
                self._termination_topic, Float32, self.termination_cb)
            self._logger.info(
                "ExecutionRunner ROS listener node subscribed to termination topic: %s" % self._termination_topic)
            self._termination_signal_received = False

        # Setup topics which should be echoed to the logger
        if echo_topics is not None:
            # Get info about all topics being published from master
            topics_info = rospy.get_published_topics()

            # Convert info from tuple-list to dictionary for easy access
            topic_to_type_dict = {}
            for topic_name, topic_type in topics_info:
                topic_to_type_dict[topic_name] = topic_type

            # Subscribe to any topics which should be echoed to the logger
            # The message type of each topic is not known ahead of time, so first have to determine the type
            self.echo_info = {}
            for topic in echo_topics:
                # Check that this topic is being published and get message type (as a string)
                topic_type = topic_to_type_dict.get(topic, None)
                if topic_type is None:
                    self._logger.warning(
                        "No publishers found for echo topic %s." % topic)
                    continue

                # Get message class from type string
                try:
                    message_class = roslib.message.get_message_class(
                        topic_type)
                except ValueError as e:
                    self._logger.warning("Error message: %s" % str(e))
                    message_class = None
                if message_class is None:
                    self._logger.warning(
                        "Failed to get message class from type string %s for topic %s" % (topic_type, topic))
                    continue

                # Setup echo topic callback, subscriber, and add to echo_info
                self._logger.info(
                    "Subscribing to topic %s with type %s" % (topic, topic_type))
                topic_cb = partial(self.echo_cb, topic)
                topic_sub = rospy.Subscriber(topic, message_class, topic_cb)
                self.echo_info[topic] = {
                    "callback": topic_cb, "subscriber": topic_sub}

    def __del__(self):
        # Unregister subscribers
        if self._termination_sub is not None:
            self._termination_sub.unregister()

        for topic_name, info_dict in viewitems(self.echo_info):
            sub = info_dict["subscriber"]
            if sub is not None:
                sub.unregister()

    def termination_cb(self, _):
        self._logger.info(
            "ExecutionRunner ROS listener node received termination signal. Setting termination flag.")
        self._termination_signal_received = True

    def echo_cb(self, topic, msg):
        self._logger.info("Received message on topic %s:" % topic)
        self._logger.info(str(msg))

    def read_termination_flag(self):
        return self._termination_signal_received
