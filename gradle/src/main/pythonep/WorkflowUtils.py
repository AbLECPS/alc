import logging
from pyxtension.streams import stream
from WorkflowDataTree import WorkflowDataTree
import threading
import base64


lock = threading.Lock()

username = "alc"
password = "alc"
user_arg = "{0}:{1}".format(username, password)

webgme_ip_address = "127.0.0.1"
webgme_host = "localhost"
webgme_port = 8888
# webgme_port = 8000
webgme_url = "http://{0}:{1}".format(webgme_host, webgme_port)

assembly_lec_setup = "Assembly_LEC_Setup"
lec_setup = "LEC_Setup"
training_data_setup = "Training_Data_Setup"

json_content_type_string = "application/json"
url_encoded_content_type_string = "application/x-www-form-urlencoded"

authorization = "{0}:{1}".format(username, password)
authorization_base64 = base64.b64encode(bytes(authorization, "utf-8")).decode("utf-8")
authorization_header_value = "Basic {0}".format(authorization_base64)

project_id_key = "projectId"
active_node_path_key = "active_node_path"
modification_key = "modifications"
set_key = "sets"
success_key = "success"

status_ok = 200
status_error = 500


webgme_router_header = {
    "Accept": "*/*",
    "Content-Type": json_content_type_string,
    "Authorization": authorization_header_value
}


def get_num_iterations(row):
    return len(row.get(WorkflowDataTree.child_id_list_key))


def max_iterations(num_iterations):
    return lambda state, workflow_data_tree, row: get_num_iterations(row) < num_iterations


def get_logger(clazz):
    logger_name = clazz.__module__ + "." + clazz.__name__
    logger = logging.Logger(logger_name)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_paths_from_results(results):
    local_results = results if isinstance(results, list) else [results]

    return stream(local_results).map(lambda result: result.get(WorkflowDataTree.path_key)).toJson()


def get_paths(job_entry):
    return [
        activity_data.get(WorkflowDataTree.path_key)
        for results_data in job_entry.get(WorkflowDataTree.data_key).values()
        for activity_data in results_data
    ]


def get_job_paths(job, local_workflow_data_tree):
    return get_paths(local_workflow_data_tree.get_entry(job.get_runtime_id()))
