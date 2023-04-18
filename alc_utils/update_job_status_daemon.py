import time
from http.client import HTTPConnection
import json
import logging
import base64
from pathlib import Path


logger = logging.Logger("update_job_status")
logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)


class Keys:
    active_node_path_key = "active_node_path"
    data_node_path_key = "data_node_path"
    data_info_key = "datainfo"
    campaign_count_key = "camp_count"
    campaign_status_file_path_key = "campaign_status_file_path"
    comment_key = "comment"
    current_working_directory_key = "current_working_directory"
    execution_directory_key = "execution_directory"
    job_id_key = "job_id"
    job_state_key = "job_state"
    job_status_key = "jobstatus"
    num_jobs_key = "num_jobs"
    password_key = "password"
    project_id_key = "projectId"
    result_dir_key = "resultDir"
    success_key = "success"
    username_key = "username"


webgme_job_info_key_set = {Keys.data_node_path_key,
                           Keys.password_key, Keys.project_id_key, Keys.username_key}

status_ok = 200

connection_wait_time = 10
failure_timeout = 2
query_delay = 5
slurm_query_jobs_url_path = "/slurm/jobs"
slurm_exec_status_file_name = "slurm_exec_status.txt"
slurm_output_file_name = "result_metadata.json"
slurm_status_file_search_pattern = "*/{0}".format(slurm_exec_status_file_name)
update_data_node_url_path = "/alcmodelupdater/updatedatanode"
update_delay_ms = 20
webgme_ip_address = "127.0.0.1"
webgme_port = 8888
# webgme_port = 8000

json_content_type_string = "application/json"
url_encoded_content_type_string = "application/x-www-form-urlencoded"

cancelled_state_string = "CANCELLED"
completed_state_string = "COMPLETED"
failed_state_string = "FAILED"
pending_state_string = "PENDING"
running_state_string = "RUNNING"

finished_status_string = "Finished"
finished_w_errors_status_string = "Finished_w_Errors"
pending_status_string = "Pending"
started_status_string = "Started"
final_status_set = {finished_status_string, finished_w_errors_status_string}

slurm_completed_job_status_set = {
    cancelled_state_string, completed_state_string, failed_state_string}

slurm_to_webgme_status_map = {
    cancelled_state_string: finished_w_errors_status_string,
    completed_state_string: finished_status_string,
    failed_state_string: finished_w_errors_status_string,
    pending_state_string: pending_status_string,
    running_state_string: started_status_string
}

slurm_headers = {
    "Accept": "*/*",
    "Content-Type": url_encoded_content_type_string
}

base_webgme_router_headers = {
    "Accept": "*/*",
    "Content-Type": json_content_type_string,
}

job_status_map = {}


def check_webgme_job_info(job_id, webgme_job_info):

    missing_key_list = []

    for key in webgme_job_info_key_set:
        if key not in webgme_job_info:
            missing_key_list.append(key)

    if len(missing_key_list) != 0:
        logger.warning("WARNING:  job {0} is missing the following keys: {1}.  Skipping.  webgme_info = {2}".format(
            job_id, missing_key_list, json.dumps(
                webgme_job_info, indent=4, sort_keys=True)
        ))
        return False
    return True


def update_data_node(webgme_job_info, modifications):

    username = webgme_job_info.get(Keys.username_key)
    password = webgme_job_info.get(Keys.password_key)
    project_id = webgme_job_info.get(Keys.project_id_key)
    active_node_path = webgme_job_info.get(Keys.data_node_path_key)

    authorization = "{0}:{1}".format(username, password)
    authorization_base64 = base64.b64encode(
        bytes(authorization, "utf-8")).decode("utf-8")
    authorization_header_value = "Basic {0}".format(authorization_base64)

    webgme_router_headers = dict(
        base_webgme_router_headers)  # COPY BASE HEADERS
    webgme_router_headers["Authorization"] = authorization_header_value

    alcmodelupdater_payload = {
        Keys.project_id_key: project_id,
        Keys.active_node_path_key: active_node_path,
        "modifications": modifications
    }
    alcmodelupdater_payload_string = json.dumps(
        alcmodelupdater_payload, indent=4, sort_keys=True)

    http_connection = HTTPConnection(webgme_ip_address, webgme_port)

    try:
        http_connection.request(
            "POST",
            str(update_data_node_url_path),
            body=alcmodelupdater_payload_string,
            headers=webgme_router_headers
        )
    except:
        logger.warning(
            "WARNING:  Problems connecting to WebGME for data node update.  Skipping.")
        return

    http_response = http_connection.getresponse()

    status = http_response.status
    output_json_string = http_response.read()

    http_connection.close()

    if status == status_ok:
        output_json = json.loads(output_json_string)
        if not output_json.get(Keys.success_key, False):
            logger.warning(
                "DATA NODE WITH PATH \"{0}\" IN PROJECT WITH ID \"{1}\" NOT UPDATED IN WEBGME MODEL.".format(
                    active_node_path, project_id
                )
            )
    else:
        logger.warning(
            "ERROR FROM WEBGME SERVER -- COULD NOT UPDATE DATA NODE WITH PATH \"{0}\" IN PROJECT WITH ID \"{1}\" "
            "IN WEBGME MODEL".format(active_node_path, project_id)
        )


def check_campaign_status(webgme_job_info, current_working_directory):

    campaign_status_file_path = webgme_job_info.get(
        Keys.campaign_status_file_path_key, None)
    if campaign_status_file_path is None:
        return

    campaign_count = webgme_job_info.get(Keys.campaign_count_key)

    campaign_status_file = Path(campaign_status_file_path)
    current_working_dir_parent = current_working_directory.absolute().parent

    slurm_exec_status_files = list(
        current_working_dir_parent.glob(slurm_status_file_search_pattern))
    num_slurm_exec_status_files = len(slurm_exec_status_files)

    if num_slurm_exec_status_files == campaign_count:
        success_counter = 0
        failed_counter = 0
        for status_file in slurm_exec_status_files:
            with status_file.open() as f:
                status_content = f.read().strip()
                if finished_w_errors_status_string in status_content:
                    failed_counter += 1
                elif finished_status_string in status_content:
                    success_counter += 1
        if success_counter == campaign_count:
            campaign_status_file.parent.mkdir(parents=True, exist_ok=True)
            with campaign_status_file.open("w") as workflow_slurm_status_fp:
                print(finished_status_string, file=workflow_slurm_status_fp)
        elif success_counter + failed_counter == campaign_count:
            campaign_status_file.parent.mkdir(parents=True, exist_ok=True)
            with campaign_status_file.open("w") as workflow_slurm_status_fp:
                print(finished_w_errors_status_string, file=workflow_slurm_status_fp)


def monitor_slurm_jobs():

    updated_completed_job_id_set = set()
    completed_job_id_set = set()

    while True:

        http_connection = HTTPConnection(webgme_ip_address, webgme_port)

        try:
            http_connection.request(
                "GET", slurm_query_jobs_url_path, headers=slurm_headers)
        except:
            logger.warning(
                "WARNING:  Problems connecting to WebGME.  Waiting {0} seconds before retry ...".format(
                    connection_wait_time
                )
            )
            time.sleep(connection_wait_time)
            continue

        http_response = http_connection.getresponse()

        output_json_string = http_response.read()

        http_connection.close()

        if http_response.status != status_ok:
            logger.error(
                "ERROR: Unable to get jobs data from slurmrestd (response: {0} {1}.  Sleeping for {2}s ...".format(
                    http_response.status, http_response.reason, failure_timeout
                )
            )
            time.sleep(failure_timeout)
            continue

        output_json_array = json.loads(output_json_string)

        # swap the contents of the set
        completed_job_id_set.clear()
        updated_completed_job_id_set, completed_job_id_set = completed_job_id_set, updated_completed_job_id_set

        for job_info in output_json_array:
            job_id = job_info.get(Keys.job_id_key)
            current_job_state = job_info.get(Keys.job_state_key)
            if current_job_state in slurm_completed_job_status_set:
                if job_id in completed_job_id_set:
                    updated_completed_job_id_set.add(job_id)
                    continue

            previous_job_status = job_status_map.get(job_id, "Initial")
            current_job_status = slurm_to_webgme_status_map.get(
                current_job_state, "Initial")

            if current_job_status != previous_job_status:
                job_status_map[job_id] = current_job_status

                json_comment = job_info.get(Keys.comment_key, "")

                try:
                    webgme_job_info = json.loads(json_comment)
                except json.decoder.JSONDecodeError as decode_error:
                    logger.warning(
                        "WARNING:  job {0} has no or malformed webgme information.  Skipping.  Value:  {1}.  "
                        "Error:  {2}".format(
                            job_id, json_comment, decode_error.msg
                        ))
                    updated_completed_job_id_set.add(job_id)
                    continue

                if not check_webgme_job_info(job_id, webgme_job_info):
                    updated_completed_job_id_set.add(job_id)
                    continue

                current_working_directory = Path(
                    job_info.get(Keys.current_working_directory_key))

                if previous_job_status == "Initial":
                    modifications = {
                        Keys.result_dir_key: str(current_working_directory)
                    }
                    logger.debug("0. debug:  job {0}, previous status = initial. new status =  {1}".format(
                        job_id, current_job_status))
                    update_data_node(webgme_job_info, modifications)

                active_node_path = webgme_job_info.get(Keys.data_node_path_key)
                project_id = webgme_job_info.get(Keys.project_id_key)

                logger.info(
                    "State of job ({0}) (active_node = \"{1}\", project_id = \"{2}\") has changed "
                    "from \"{3}\" to \"{4}\"".format(
                        job_id, active_node_path, project_id, previous_job_status, current_job_status
                    )
                )

                modifications = {
                    Keys.job_status_key: current_job_status
                }
                update_data_node(webgme_job_info, modifications)

                slurm_exec_status_file_path = Path(
                    current_working_directory, slurm_exec_status_file_name)
                with slurm_exec_status_file_path.open("w") as slurm_exec_status_fp:
                    print(current_job_status, file=slurm_exec_status_fp, end=None)

                if current_job_status in final_status_set:
                    logger.info(
                        "Final state of job ({0}) (active_node = \"{1}\", project_id = \"{2}\") is \"{3}\"".format(
                            job_id, active_node_path, project_id, current_job_status
                        )
                    )

                    if current_job_status == finished_status_string:

                        slurm_output_file_path = Path(
                            current_working_directory, slurm_output_file_name)

                        if slurm_output_file_path.exists():
                            with slurm_output_file_path.open("r") as slurm_output_file_fp:
                                slurm_output_json = json.load(
                                    slurm_output_file_fp)
                            if isinstance(slurm_output_json, list) and len(slurm_output_json) == 1:
                                slurm_output_json = slurm_output_json[0]
                            slurm_output_json_string = json.dumps(
                                slurm_output_json, indent=4, sort_keys=True)

                            modifications = {
                                Keys.data_info_key: slurm_output_json_string
                            }
                            update_data_node(webgme_job_info, modifications)

                    updated_completed_job_id_set.add(job_id)
                    del job_status_map[job_id]
                    check_campaign_status(
                        webgme_job_info, current_working_directory)

                # DELAY IN CASE A FLOOD OF UPDATES (UNLIKELY)
                time.sleep(update_delay_ms / 1000)

        time.sleep(query_delay)


if __name__ == "__main__":
    monitor_slurm_jobs()
