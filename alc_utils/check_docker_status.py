#!/usr/bin/env python2
import docker
import docker.errors
import errno

DOCKER_WAIT_TIMEOUT_S = 1
DOCKER_STOP_TIMEOUT_S = 10
DOCKER_TIMEOUT_BUFFER_S = 60


class CheckDockerStatus:
    """This class configures a docker environment and executes an experiment based on the provided execution file."""

    def __init__(self, docker_container_name, timeout_buffer=DOCKER_TIMEOUT_BUFFER_S):
        self.docker_container_name = docker_container_name
        self._timeout_buffer = timeout_buffer
        self.docker_client = docker.from_env()
        self.docker_api_client = docker.APIClient(
            base_url='unix://var/run/docker.sock')

    # returns -1,"" if the container is not found
    # returns 1, "running" if the container is running
    # returns 0, state if the container is another state
    def check_status(self, logger):

        try:
            container = self.docker_client.containers.get(
                self.docker_container_name)
        except docker.errors.NotFound as exc:
            self.dump_message(logger, 1, 'Container named %s not found' % (
                self.docker_container_name))
            return -1, ""

        container_state = container.attrs["State"]
        is_running = container_state["Status"] == "running"
        return is_running, container_state

    def terminate_docker(self, logger):

        container = ""

        try:
            container = self.docker_client.containers.get(
                self.docker_container_name)
        except docker.errors.NotFound as exc:
            self.dump_message(logger, 1, 'Container named %s not found' % (
                self.docker_container_name))
            return -1

        try:
            container.stop(timeout=DOCKER_STOP_TIMEOUT_S)
            self.dump_message(logger, 0, 'Container stopped successfully.')
        except docker.errors.NotFound as e:
            error_encountered = True

            self.dump_message(logger, 1, 'Docker APIError occured with status code %d when stopping container %s'
                              % (e.status_code, container.id))
            self.dump_message(logger, 1, 'APIError message: %s' % e.message)

            return -1

        try:
            self.dump_message(logger, 0, "Removing container '%s'..." %
                              self.docker_container_name)
            container.remove()
            self.dump_message(logger, 0, 'Container removed successfully.')
        except docker.errors.APIError as e:
            error_encountered = True
            self.dump_message(logger, 1, 'Docker APIError occured with status code %d when removing container %s'
                              % (e.status_code, container.id))
            self.dump_message(logger, 1, 'APIError message: %s' % e.message)
            return -1

        return 0

    def dump_message(self, logger, iserror, message):
        if logger:
            if (iserror):
                logger.error(message)
            else:
                logger.info(message)
        else:
            print(message)
