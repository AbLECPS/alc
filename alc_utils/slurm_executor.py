from pathlib import Path
import os
import json
import copy
import uuid
import time
from http.client import HTTPConnection
import alc_utils.config as alc_config
import alc_utils.common as alc_common

from .update_job_status_daemon import Keys as UpdateKeys


json_content_type_string = "application/json"
url_encoded_content_type_string = "application/x-www-form-urlencoded"

JOB_TYPE_DEFAULTS = {
    "experimentsetup": alc_config.slurm_execution_defaults.EXPERIMENT_DEFAULT_JOB_INFO,
    "default": alc_config.slurm_execution_defaults.DEFAULT_JOB_INFO
}

status_ok = 200


class SlurmParameterKeys:
    job_name_key = "name"
    number_of_tasks_key = "tasks"
    standard_output_key = "standard_output"
    time_limit_key = "time_limit"


class SlurmSubmissionKeys:
    job_key = "job"
    script_key = "script"


class WebGMEKeys:
    command_for_srun_key = "command_for_srun"
    environment_key = "environment"
    job_name_key = "job_name"
    job_type_key = "jobtype"
    job_uuid_key = "job_uuid"
    number_of_tasks_key = "ntasks"
    project_name_key = "name"
    project_owner_key = "owner"
    result_dir_key = "result_dir"
    standard_output_key = "output"
    task_file_name_key = "task_file_name"
    time_limit_key = "time"
    webgme_port_key = "webgme_port"
    workflow_output_key = 'wf_output_path'
    repo_home_key = 'REPO_HOME'


slurm_submit_job_url_path = Path("/slurm/job/submit")
slurm_query_jobs_url_path = Path("/slurm/jobs")
localhost_ip_address = "127.0.0.1"

job_id_from_jobs_wait_period = 3
max_jobs_data_iterations = 10
max_submission_tries = 5
jobs_data_timeout = 2

job_info_key_map = {
    UpdateKeys.campaign_count_key: None,
    WebGMEKeys.command_for_srun_key: None,
    UpdateKeys.data_node_path_key: None,
    WebGMEKeys.job_name_key: SlurmParameterKeys.job_name_key,
    WebGMEKeys.number_of_tasks_key: SlurmParameterKeys.number_of_tasks_key,
    WebGMEKeys.project_name_key: None,
    WebGMEKeys.project_owner_key: None,
    WebGMEKeys.result_dir_key: None,
    SlurmSubmissionKeys.script_key: None,
    WebGMEKeys.standard_output_key: SlurmParameterKeys.standard_output_key,
    WebGMEKeys.task_file_name_key: None,
    WebGMEKeys.time_limit_key: SlurmParameterKeys.time_limit_key,
    WebGMEKeys.workflow_output_key: None
}


def is_job_submitted(job_uuid, slurm_port):
    headers = {
        "Accept": "*/*",
        "Content-Type": url_encoded_content_type_string
    }

    time.sleep(job_id_from_jobs_wait_period)

    output_json_string = "{}"
    jobs_data_iterations = 0
    jobs_data_acquired = False
    while not jobs_data_acquired and jobs_data_iterations < max_jobs_data_iterations:

        jobs_data_iterations += 1
        print(
            "Attempting to check if slurm job (workflow id {0}) submitted to slurmrestd (attempt {1} of {2}) ...".
            format(job_uuid, jobs_data_iterations, max_jobs_data_iterations)
        )

        http_connection = HTTPConnection(localhost_ip_address, slurm_port)
        http_connection.request("GET", str(
            slurm_query_jobs_url_path), headers=headers)

        http_response = http_connection.getresponse()

        output_json_string = http_response.read()

        http_connection.close()

        if http_response.status == 200:
            jobs_data_acquired = True
        else:
            jobs_data_iterations += 1
            print(
                "ERROR: Unable to get jobs data (workflow id {0}) from slurmrestd.  Sleeping for {1}s ...".format(
                    job_uuid, jobs_data_timeout
                )
            )
            time.sleep(jobs_data_timeout)

    if not jobs_data_acquired:
        message = (
            "ERROR: Unable to check if slurm job (workflow id {0}) submitted to slurmrestd.  " +
            "Maximum attempts ({1}) exceeded.  Here's hoping it got submitted."
        ).format(job_uuid, max_jobs_data_iterations)
        print(message)
        return True

    output_json_array = json.loads(output_json_string)

    for output_json in output_json_array:
        comment_json_string = output_json.get(UpdateKeys.comment_key)
        comment_json = json.loads(comment_json_string)
        if comment_json.get(WebGMEKeys.job_uuid_key) == job_uuid:
            return True

    message = "WARNING: slurm job (workflow id {0}) not submitted, trying again".format(
        job_uuid)
    print(message)

    return False


def submit_slurm_job(payload_string, slurm_port, job_uuid):

    print("Using slurm REST api payload: {0}".format(payload_string))

    headers = {
        "Accept": "*/*",
        "Content-Type": json_content_type_string,
    }

    submission_tries = 0

    while submission_tries < max_submission_tries:

        submission_tries += 1

        http_connection = HTTPConnection(localhost_ip_address, slurm_port)

        http_connection.request("POST", str(
            slurm_submit_job_url_path), body=payload_string, headers=headers)

        http_response = http_connection.getresponse()

        status = http_response.status
        output_json_string = http_response.read()

        http_connection.close()

        if status == status_ok:
            print(
                "Job (workflow id {0}) was successfully submitted.".format(job_uuid))
            print("Received response from POST: {0}".format(
                output_json_string))
            return

        output_json = json.loads(output_json_string)
        output_json_string = json.dumps(output_json, indent=4, sort_keys=True)

        print(
            "ERROR: submitting job (workflow id {0}) to slurmrestd, status = {1}, {2}.  "
            "Unable to ascertain if job was submitted.".format(
                job_uuid, status, output_json_string)
        )
        print("Attempting to see if job (workflow id {0}) was submitted from jobs data".format(
            job_uuid))

        if is_job_submitted(job_uuid, slurm_port):
            print(
                "Job (workflow id {0}) was successfully submitted.".format(job_uuid))
            return

    print("ERROR:  Unable to submit job (workflow id {0})".format(job_uuid))


def slurm_deploy_job(relative_job_dir, job_params=None):
    # Move to job directory
    job_dir_local_path = Path(alc_config.WORKING_DIRECTORY, relative_job_dir)
    os.chdir(str(job_dir_local_path))

    # Make sure all parameter names are lowercase
    if job_params is None:
        job_params = {}
    job_params = alc_common.dict_convert_key_case(
        job_params, desired_case="lower")

    # COPY ALL ENVIRONMENT VARIABLES TO SLURM
    environment_map = {name: os.environ[name] for name in os.environ}
    if WebGMEKeys.repo_home_key in job_params and job_params.get(WebGMEKeys.repo_home_key, None):
        environment_map[WebGMEKeys.repo_home_key] = job_params.get(
            WebGMEKeys.repo_home_key)
        job_params.pop(WebGMEKeys.repo_home_key)
        print("came here in slurm deploy")
        print(str(job_params))
        #environment_map['ALC_HOME']= job_params.get(WebGMEKeys.repo_home_key)

    # Fill out job info. If the job type is recognized, use specific defaults. Otherwise, use generic defaults.
    # Update default values with passed parameters
    job_type = job_params.pop(WebGMEKeys.job_type_key, "default").lower()
    raw_job_info = copy.copy(JOB_TYPE_DEFAULTS.get(
        job_type, JOB_TYPE_DEFAULTS["default"]))
    raw_job_info.update(job_params)

    #
    # COLLECT SLURM PARAMETERS
    #
    slurm_parameter_map = {}
    for key, value in raw_job_info.items():
        new_key = job_info_key_map.get(key, key)
        if new_key is not None:
            slurm_parameter_map[new_key] = value

    slurm_parameter_map[UpdateKeys.current_working_directory_key] = str(
        job_dir_local_path)
    slurm_parameter_map[WebGMEKeys.environment_key] = environment_map

    slurm_port = slurm_parameter_map.pop(WebGMEKeys.webgme_port_key, 8888)

    job_uuid = uuid.uuid4().hex

    if UpdateKeys.data_node_path_key in job_params:
        webgme_info = {
            UpdateKeys.username_key: "alc",
            UpdateKeys.password_key: "alc",
            UpdateKeys.project_id_key: "{0}+{1}".format(
                job_params[WebGMEKeys.project_owner_key], job_params[WebGMEKeys.project_name_key]
            ),
            UpdateKeys.data_node_path_key: job_params[UpdateKeys.data_node_path_key],
            WebGMEKeys.job_uuid_key: job_uuid
        }
        if WebGMEKeys.workflow_output_key in job_params:
            webgme_info[UpdateKeys.campaign_status_file_path_key] = job_params[WebGMEKeys.workflow_output_key]
            webgme_info[UpdateKeys.campaign_count_key] = job_params[UpdateKeys.campaign_count_key]

        webgme_info_string = json.dumps(webgme_info, indent=4, sort_keys=True)

        slurm_parameter_map[UpdateKeys.comment_key] = webgme_info_string

    #
    # CONSTRUCT SCRIPT FOR SLURM
    #
    script_string = """\
#!/bin/sh
set -e
srun {0}""".format(raw_job_info.get(WebGMEKeys.command_for_srun_key))

    #
    # CONSTRUCT SLURM REST API PAYLOAD
    #
    slurm_payload = {
        SlurmSubmissionKeys.job_key: slurm_parameter_map,
        SlurmSubmissionKeys.script_key: script_string
    }

    slurm_payload_string = json.dumps(slurm_payload, indent=4, sort_keys=True)

    submit_slurm_job(slurm_payload_string, slurm_port, job_uuid)

    print("Submitted job (workflow id {0})".format(job_uuid))
