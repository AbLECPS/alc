import logging
from pyxtension.streams import stream
from WorkflowDataTree import WorkflowDataTree


username = "alc"
password = "alc"
user_arg = "{0}:{1}".format(username, password)

webgme_host = "localhost"
webgme_port = 8888
webgme_url = "http://{0}:{1}".format(webgme_host, webgme_port)

assembly_lec_setup = "Assembly_LEC_Setup"
lec_setup = "LEC_Setup"
training_data_setup = "Training_Data_Setup"


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
