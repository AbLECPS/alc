import threading
import uuid
import string
from addict import Dict
import shutil
from pathlib import Path
from WorkflowDataTree import WorkflowDataTree
from BaseOperation import BaseOperation
import WorkflowUtils


class BaseJob(BaseOperation):

    _id_file_name = "id_file"
    _hex_character_set = set(string.hexdigits)
    _logger = None
    _parameter_updates_key = "parameter_updates"
    _parameter_filter_key = "parameter_filter"

    def __init__(self, job_name, previous_job_name_list, next_job_name_list):
        BaseOperation.__init__(self, previous_job_name_list, next_job_name_list)
        self.job_name = job_name
        self.runtime_id_map = {}
        self.is_root = False
        self.static_parameters = {}

    def get_job_name(self):
        return self.job_name

    def execute(self, state, workflow_data_tree, execution_parameters, directory, parent_iteration_row):
        return state, workflow_data_tree, {}

    def set_is_root(self):
        self.is_root = True

    def set_parameter_updates(self, parameter_updates):
        if not isinstance(parameter_updates, dict) and not callable(parameter_updates):
            BaseJob._logger.warning(
                "ERROR: WorkflowJob.set_parameter_updates method: argument must be a dict "
                "object or a function that returns a dict object.  Parameter updates rejected."
            )
            return

        self.static_parameters[self._parameter_updates_key] = parameter_updates

        return self

    def set_parameter_filter(self, parameter_filter):

        if not isinstance(parameter_filter, list) and not isinstance(parameter_filter, set):
            BaseJob._logger.warning(
                "ERROR: WorkflowJob.set_parameter_filter method: argument must be a list or set.  "
                "Parameter filter rejected."
            )
            return

        self.static_parameters[self._parameter_filter_key] = parameter_filter

        return self

    def get_updated_execution_parameters(self, workflow_data, execution_parameters):

        if self._parameter_updates_key not in self.static_parameters:
            return execution_parameters

        all_parameters = Dict(execution_parameters)

        parameter_updates = self.static_parameters.get(self._parameter_updates_key, Dict())
        if callable(parameter_updates):
            parameter_updates = parameter_updates(workflow_data)

        all_parameters.update(parameter_updates)

        return all_parameters

    def filter_parameters(self, execution_parameters):

        if self._parameter_filter_key not in self.static_parameters:
            return execution_parameters

        filter_parameter_list = self.static_parameters.get(self._parameter_filter_key)
        output_parameters = {key: execution_parameters.get(key) for key in filter_parameter_list}

        return Dict(output_parameters)

    def get_unique_id(self):
        return WorkflowDataTree.root_key if self.is_root else uuid.uuid4().hex

    def is_valid_id(self, id_string):
        if self.is_root:
            return id_string == WorkflowDataTree.root_key

        for character in id_string:
            if character not in BaseJob._hex_character_set:
                return False

        return True

    def get_dynamic_id(self, directory):
        id_file_path = Path(directory, BaseJob._id_file_name)

        if id_file_path.is_file():
            with id_file_path.open("r") as id_fp:
                id_value = id_fp.read().strip()
            if id_value and self.is_valid_id(id_value):
                return id_value

        if id_file_path.exists():
            if id_file_path.is_dir():
                shutil.rmtree(id_file_path)
            else:
                id_file_path.unlink()

        id_string = self.get_unique_id()

        directory.mkdir(parents=True, exist_ok=True)
        with Path(directory, BaseJob._id_file_name).open("w") as id_fp:
            print(id_string, file=id_fp)

        return id_string

    def set_runtime_id(self, directory):
        self.runtime_id_map[threading.get_ident()] = self.get_dynamic_id(directory)

    def get_runtime_id(self):
        return self.runtime_id_map.get(threading.get_ident())


BaseJob._logger = WorkflowUtils.get_logger(BaseJob)
