import os
from pathlib import Path
import json
import time
import uuid
import WorkflowUtils
from http.client import HTTPConnection
import inotify.adapters
import inotify.constants
import logging
from BaseTask import BaseTask
from ProjectParameters import ProjectParameters


class SlurmTask(BaseTask):
    logger = None

    username = "alc"
    password = "alc"

    comment_key = "comment"
    cpus_per_task_key = "cpus_per_task"
    current_working_directory_key = "current_working_directory"
    data_node_path_key = "data_node_path"
    environment_key = "environment"
    gres_key = "gres"
    job_key = "job"
    job_uuid_key = "job_uuid"
    name_key = "name"
    node_path_key = "node_path"
    nodes_key = "nodes"
    partition_key = "partition"
    password_key = "password"
    project_id_key = "projectId"
    script_key = "script"
    standard_error_key = "standard_error"
    standard_input_key = "standard_input"
    standard_output_key = "standard_output"
    tasks_key = "tasks"
    time_limit_key = "time_limit"
    username_key = "username"
    working_directory_key = "current_working_directory"

    create_data_node_url_path = "/alcmodelupdater/createdatanode"

    slurm_query_jobs_url_path = "/slurm/jobs"
    slurm_submit_job_url_path = "/slurm/job/submit"

    finished_status_string = "Finished"
    finished_w_errors_status_string = "Finished_w_Errors"
    completed_status_string_set = {finished_status_string, finished_w_errors_status_string}

    submit_job_parameters = {
        cpus_per_task_key: 4,
        gres_key: "gpu:1",
        nodes_key: 1,
        partition_key: "primary",
        tasks_key: 1,
        time_limit_key: 47
    }

    slurm_exec_status_file_name = "slurm_exec_status.txt"
    output_file_name = "result_metadata.json"
    output_file_exists_iterations = 5
    output_file_exists_timeout = 2
    max_bad_request_count = 10
    max_submission_tries = 5
    job_id_from_jobs_wait_period = 3
    max_jobs_data_iterations = 10
    jobs_data_timeout = 2

    initial_poll_interval = 2
    poll_interval = 10

    def __init__(
            self,
            active_node=None,
            execution_name=None,
            working_dir=Path("."),
            command=None,
            standard_input=None,
            standard_output=None,
            standard_error=None
    ):
        if command is None:
            command = []

        self.active_node = active_node
        self.execution_name = execution_name
        self.data_node = None
        self.success = True

        BaseTask.__init__(
            self,
            working_dir=working_dir,
            command=command,
            standard_input=standard_input,
            standard_output=standard_output,
            standard_error=standard_error
        )

    def _is_job_submitted(self, job_uuid):
        headers = {
            "Accept": "*/*",
            "Content-Type": WorkflowUtils.url_encoded_content_type_string
        }

        time.sleep(self.job_id_from_jobs_wait_period)

        output_json_string = "{}"
        jobs_data_iterations = 0
        jobs_data_acquired = False
        while not jobs_data_acquired and jobs_data_iterations < self.max_jobs_data_iterations:

            jobs_data_iterations += 1
            SlurmTask.logger.info(
                "Attempting to check if slurm job (workflow id {0}) submitted to slurmrestd "
                "(attempt {1} of {2}) ...".format(job_uuid, jobs_data_iterations, self.max_jobs_data_iterations)
            )

            http_connection = HTTPConnection(WorkflowUtils.webgme_ip_address, WorkflowUtils.webgme_port)
            http_connection.request("GET", self.slurm_query_jobs_url_path, headers=headers)

            http_response = http_connection.getresponse()

            output_json_string = http_response.read()

            http_connection.close()

            if http_response.status == 200:
                jobs_data_acquired = True
            else:
                jobs_data_iterations += 1
                SlurmTask.logger.warning(
                    "ERROR: Unable to get jobs data (workflow id {0}) from slurmrestd.  Sleeping for {1}s ...".format(
                        job_uuid, self.jobs_data_timeout
                    )
                )
                time.sleep(self.jobs_data_timeout)

        if not jobs_data_acquired:
            message = (
                    "ERROR: Unable to check if slurm job (workflow id {0}) submitted to slurmrestd.  " +
                    "Maximum attempts ({1}) exceeded.  Here's hoping it got submitted."
            ).format(job_uuid, self.max_jobs_data_iterations)
            SlurmTask.logger.warning(message)
            return True

        output_json_array = json.loads(output_json_string)

        for output_json in output_json_array:
            comment_json_string = output_json.get(self.comment_key)
            comment_json = json.loads(comment_json_string)
            if comment_json.get(self.job_uuid_key) == job_uuid:
                return True

        message = "WARNING: slurm job (workflow id {0}) not submitted, trying again".format(job_uuid)
        SlurmTask.logger.warning(message)

        return False

    def _create_data_node(self):

        alcmodelupdater_payload = {
            WorkflowUtils.project_id_key: ProjectParameters.get_project_id(),
            WorkflowUtils.active_node_path_key: self.active_node,
            self.name_key: self.execution_name,
            WorkflowUtils.modification_key: {},
            WorkflowUtils.set_key: {}
        }

        http_connection = HTTPConnection(WorkflowUtils.webgme_ip_address, WorkflowUtils.webgme_port)

        alcmodelupdater_payload_string = json.dumps(alcmodelupdater_payload, indent=4, sort_keys=True)

        http_connection.request(
            "POST",
            str(self.create_data_node_url_path),
            body=alcmodelupdater_payload_string,
            headers=WorkflowUtils.webgme_router_header
        )

        http_response = http_connection.getresponse()

        status = http_response.status
        output_json_string = http_response.read()

        http_connection.close()

        if status == WorkflowUtils.status_ok:
            output_json = json.loads(output_json_string)
            if self.node_path_key in output_json:
                self.data_node = output_json.get(self.node_path_key)
            else:
                self.logger.warning(
                    "DATA NODE NOT CREATED IN WEBGME MODEL.  Will not be able to show status of slurm"
                    " job or present results."
                )
        else:
            self.logger.warning(
                "ERROR FROM WEBGME SERVER -- COULD NOT CREATE DATA NODE IN WEBGME MODEL.   Will not be able "
                "to show status of slurm job or present results."
            )

    def get_webgme_info(self):

        return {
            self.username_key: self.username,
            self.password_key: self.password,
            self.project_id_key: ProjectParameters.get_project_id(),
            self.data_node_path_key: self.data_node,
            self.job_uuid_key: uuid.uuid4().hex
        }

    def _submit_slurm_job(self):

        self._create_data_node()

        script_string = """\
#!/bin/sh
set -e
srun {0}""".format(" ".join(self.command))

        webgme_info = self.get_webgme_info()

        slurm_payload = {
            self.job_key: self.get_slurm_submit_job_parameters(webgme_info),
            self.script_key: script_string
        }
        slurm_payload_string = json.dumps(slurm_payload, indent=4, sort_keys=True)

        headers = {
            "Accept": "*/*",
            "Content-Type": WorkflowUtils.json_content_type_string  # ,
        }
        # print(SlurmTask.slurm_port)

        submission_tries = 0

        job_uuid = webgme_info.get(self.job_uuid_key)
        while submission_tries < self.max_submission_tries:
            submission_tries += 1

            http_connection = HTTPConnection(WorkflowUtils.webgme_ip_address, WorkflowUtils.webgme_port)

            http_connection.request(
                "POST", str(SlurmTask.slurm_submit_job_url_path), body=slurm_payload_string, headers=headers
            )

            http_response = http_connection.getresponse()

            status = http_response.status
            reason = http_response.reason
            output_json_string = http_response.read()

            http_connection.close()

            if status == WorkflowUtils.status_ok:
                message = "Job (workflow id {0}) successfully submitted".format(
                    job_uuid, status, reason
                )
                SlurmTask.logger.info(message)
                return

            output_json = json.loads(output_json_string)
            output_json_string = json.dumps(output_json, indent=4, sort_keys=True)

            message = "ERROR: submitting job (workflow id {0}) to slurmrestd, status = {1}, {2}.  " \
                      "Unable to ascertain if job was submitted.".format(job_uuid, status, output_json_string)
            SlurmTask.logger.warning(message)

            SlurmTask.logger("Attempting to see if job (workflow id {0}) was submitted from jobs data".format(job_uuid))

            if self._is_job_submitted(job_uuid):
                SlurmTask.logger.info("Job (workflow id {0}) was successfully submitted.".format(job_uuid))
                return

        SlurmTask.logger.warning("ERROR:  Unable to submit job (workflow id {0})".format(job_uuid))

    def _slurm_wait(self):

        slurm_exec_status_file_path = Path(self.working_dir, self.slurm_exec_status_file_name)

        # SET UP TO WAIT FOR EXECUTION STATUS FILE TO BE CREATED
        slurm_exec_status_file_parent_path = slurm_exec_status_file_path.parent
        slurm_exec_status_file_watcher = inotify.adapters.Inotify()
        file_wd = slurm_exec_status_file_watcher.add_watch(
            str(slurm_exec_status_file_parent_path), inotify.constants.IN_CREATE
        )

        if not slurm_exec_status_file_path.exists():
            for _ in slurm_exec_status_file_watcher.event_gen(yield_nones=False):
                if slurm_exec_status_file_path.exists():
                    break

        slurm_exec_status_file_watcher.remove_watch_with_id(file_wd)

        # SET UP TO WAIT UNTIL COMPLETION STRING WRITTEN TO slurm_exec_status_file FILE
        file_wd = slurm_exec_status_file_watcher.add_watch(
            str(slurm_exec_status_file_path), inotify.constants.IN_CLOSE_WRITE
        )

        # WAIT FOR SOMETHING TO BE WRITTEN TO THE EXECUTION FILE
        with slurm_exec_status_file_path.open("r") as execution_file_fp:
            status_string = execution_file_fp.read().strip()

        while status_string not in self.completed_status_string_set:
            next(slurm_exec_status_file_watcher.event_gen(yield_nones=False))
            with slurm_exec_status_file_path.open("r") as execution_file_fp:
                status_string = execution_file_fp.read().strip()

        slurm_exec_status_file_watcher.remove_watch_with_id(file_wd)

    def get_success(self):
        return self.success

    def get_data_node(self):
        return self.data_node

    def execute(self):

        self._submit_slurm_job()
        self._slurm_wait()

        return self

    def set_standard_input(self, standard_input):
        if standard_input is not None and not isinstance(standard_input, str) and not isinstance(standard_input, Path):
            raise Exception("Must use file name (not file object) for standard input")
        return BaseTask.set_standard_input(self, standard_input)

    def set_standard_output(self, standard_output):
        if standard_output is not None \
                and not isinstance(standard_output, str) and not isinstance(standard_output, Path):
            raise Exception("Must use file name (not file object) for standard output")
        return BaseTask.set_standard_output(self, standard_output)

    def set_standard_error(self, standard_error):
        if standard_error is not None and not isinstance(standard_error, str) and not isinstance(standard_error, Path):
            raise Exception("Must use file name (not file object) for standard error")
        return BaseTask.set_standard_error(self, standard_error)

    def get_slurm_submit_job_parameters(self, webgme_info):

        slurm_submit_job_parameters = dict(self.submit_job_parameters)

        webgme_info_string = json.dumps(webgme_info, indent=4, sort_keys=True)
        slurm_submit_job_parameters[self.comment_key] = webgme_info_string

        if self.standard_input is not None:
            slurm_submit_job_parameters[self.standard_input_key] = str(self.standard_input)

        if self.standard_output is not None:
            slurm_submit_job_parameters[self.standard_output_key] = str(self.standard_output)

        if self.standard_error is not None:
            slurm_submit_job_parameters[self.standard_error_key] = str(self.standard_error)

        slurm_submit_job_parameters[self.working_directory_key] = str(self.working_dir)

        environment_map = {name: os.environ[name] for name in os.environ}
        slurm_submit_job_parameters[self.environment_key] = environment_map

        return slurm_submit_job_parameters


SlurmTask.logger = WorkflowUtils.get_logger(SlurmTask)
SlurmTask.logger.setLevel(logging.INFO)
