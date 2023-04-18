import threading
import uuid
import string
import shutil
from pathlib import Path
from WorkflowDataTree import WorkflowDataTree


class BaseJob:

    _id_file_name = "id_file"
    _hex_character_set = set(string.hexdigits)

    def __init__(self, parent_loop, job_name, previous_job_name_list, next_job_name_list):
        self.parent_loop = parent_loop
        self.job_name = job_name
        self.previous_job_name_set = set(previous_job_name_list)
        self.next_job_name_set = set(next_job_name_list)
        self.path = "{0}.{1}".format(parent_loop.path if parent_loop is not None else "root", job_name)
        self.runtime_id_map = {}
        self.is_root = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def get_job_name(self):
        return self.job_name

    def get_previous_job_name_set(self):
        return set(self.previous_job_name_set)

    def get_next_job_name_set(self):
        return set(self.next_job_name_set)

    def execute(self, state, workflow_data_tree, parameters, directory, parent_iteration_row):
        return state, workflow_data_tree, {}

    def set_is_root(self):
        self.is_root = True

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
                id_value = id_fp.read()
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
