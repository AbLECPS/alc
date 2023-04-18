from pathlib import Path
from addict import Dict
import operator
import json
from pyxtension.streams import stream
from Result import Result


class WorkflowDataTree:

    child_id_list_key = "child_id_list"
    data_key = "data"
    directory_key = "directory"
    id_key = "id"
    index_key = "index"
    info_key = "info"
    name_key = "name"
    parent_id_key = "parent_id"
    path_key = "path"
    result_url_key = "result_url"
    root_key = "root"
    type_key = "type"

    max_iteration_specifier = "max"
    all_iterations_specifier = "all"
    valid_iteration_specifier_set = {max_iteration_specifier, all_iterations_specifier}
    default_iterations_specifier = all_iterations_specifier

    iteration_type = "ITERATION"
    job_type = "JOB"
    loop_type = "LOOP"

    def __init__(self):
        self.data_map = Dict()

    @staticmethod
    def stream_data(data):
        return stream(data)

    def copy(self):
        workflow_data_tree = WorkflowDataTree()
        workflow_data_tree.data_map = Dict(self.data_map)
        return workflow_data_tree

    def get_data(self):
        return list(self.data_map.values())

    def load_from_file(self, input_file):
        actual_input_file = input_file
        if isinstance(actual_input_file, str):
            actual_input_file = Path(actual_input_file)

        with actual_input_file.open("r") as input_fp:
            self.data_map = json.load(input_fp, object_hook=lambda x: Dict(x))

    def save_to_file(self, output_file):
        actual_output_file = output_file
        if isinstance(actual_output_file, str):
            actual_output_file = Path(actual_output_file)

        with actual_output_file.open("w") as output_fp:
            json.dump(self.data_map, output_fp, indent=4, sort_keys=True)

    def add_job_data(self, job, job_data, parent_iteration_id):

        job_id = job.get_runtime_id()

        new_row = Dict({
            WorkflowDataTree.type_key: WorkflowDataTree.job_type,
            WorkflowDataTree.name_key: job.job_name,
            WorkflowDataTree.parent_id_key: parent_iteration_id,
            WorkflowDataTree.id_key: job_id,
            WorkflowDataTree.data_key: job_data,
            WorkflowDataTree.child_id_list_key: []
        })

        self.data_map[job_id] = new_row

        if parent_iteration_id in self.data_map:
            self.data_map.get(parent_iteration_id)[WorkflowDataTree.child_id_list_key].append(job_id)

        return new_row

    def add_iteration_data(self, iteration_name, iteration_id, iteration_index, parent_loop_id):
        new_row = Dict({
            WorkflowDataTree.type_key: WorkflowDataTree.iteration_type,
            WorkflowDataTree.name_key: iteration_name,
            WorkflowDataTree.id_key: iteration_id,
            WorkflowDataTree.parent_id_key: parent_loop_id,
            WorkflowDataTree.index_key: iteration_index,
            WorkflowDataTree.child_id_list_key: []
        })

        self.data_map[iteration_id] = new_row

        if parent_loop_id in self.data_map:
            self.data_map.get(parent_loop_id)[WorkflowDataTree.child_id_list_key].append(iteration_id)

        return new_row

    def add_loop_data(self, job, parent_iteration_id):

        loop_id = job.get_runtime_id()

        new_row = Dict({
            WorkflowDataTree.type_key: WorkflowDataTree.loop_type,
            WorkflowDataTree.name_key: job.job_name,
            WorkflowDataTree.id_key: loop_id,
            WorkflowDataTree.parent_id_key: parent_iteration_id,
            WorkflowDataTree.child_id_list_key: []
        })

        self.data_map[loop_id] = new_row

        if parent_iteration_id in self.data_map:
            self.data_map.get(parent_iteration_id)[WorkflowDataTree.child_id_list_key].append(loop_id)

        return new_row

    def get_entry(self, item_id):
        return self.data_map.get(item_id, None)

    def merge_data(self, other):
        self_keys = set(self.data_map.keys())
        other_keys = set(other.data_map.keys())

        common_keys = self_keys.intersection(other_keys)
        for key in common_keys:
            value = self.data_map.get(key)
            if WorkflowDataTree.child_id_list_key in value:
                value[WorkflowDataTree.child_id_list_key] = list(
                    set(value.get(WorkflowDataTree.child_id_list_key)).union(
                        set(other.data_map.get(key).get(WorkflowDataTree.child_id_list_key))
                    )
                )

        other_unique_keys = other_keys.difference(self_keys)
        for key in other_unique_keys:
            self.data_map[key] = other.data_map.get(key)

        return self

    @staticmethod
    def extend_and_return(list1, list2):
        list1.extend(list2)
        return list1

    def get_leaf_jobs(self, row):

        if row.get(WorkflowDataTree.type_key) == WorkflowDataTree.job_type:
            return [row]

        return stream(self.get_child_rows(row)) \
            .map(lambda child_row: self.get_leaf_jobs(child_row)) \
            .reduce(lambda item1, item2: WorkflowDataTree.extend_and_return(item1, item2))

    @staticmethod
    def get_result_path(result):
        return result.get(WorkflowDataTree.path_key)

    def get_child_rows(self, item):
        return [self.get_entry(child_id) for child_id in item.get(WorkflowDataTree.child_id_list_key)]

    def get_parent_row(self, item):
        parent_id = item.get(WorkflowDataTree.parent_id_key)
        return None if parent_id is None else self.get_entry(parent_id)

    @staticmethod
    def get_row_type(row):
        return row.get(WorkflowDataTree.type_key)

    @staticmethod
    def is_iteration(row):
        return WorkflowDataTree.get_row_type(row) == WorkflowDataTree.iteration_type

    def get_iteration_num(self, row):
        iteration_row = \
            row if WorkflowDataTree.is_iteration(row) else self.get_parent_row(row)
        return int(iteration_row.get(WorkflowDataTree.index_key))

    @staticmethod
    def get_num_iterations(row):
        return len(row.get(WorkflowDataTree.child_id_list_key))

    def get_all_child_jobs(self, row):
        if row.get(WorkflowDataTree.type_key) == WorkflowDataTree.job_type:
            return [row]

        return stream(self.get_child_rows(row)) \
            .flatMap(lambda item: self.get_child_rows(item)).toList()

    def get_child_iteration_list(self, row, iteration_specifier):

        if iteration_specifier == WorkflowDataTree.all_iterations_specifier:
            return self.get_child_rows(row)

        if iteration_specifier == WorkflowDataTree.max_iteration_specifier:
            iteration_specifier = -1

        child_iteration_id_list = row.get(WorkflowDataTree.child_id_list_key)

        comparator = operator.lt if iteration_specifier >= 0 else operator.le

        if comparator(abs(iteration_specifier), len(child_iteration_id_list)):
            return [self.get_entry(child_iteration_id_list[iteration_specifier])]

        return []

    def get_path_data_relative(self, data_path, initial_iteration_list):

        data_list = initial_iteration_list

        for item in data_path:
            data = stream(data_list)
            iteration_specifier = WorkflowDataTree.default_iterations_specifier
            if isinstance(item, list):
                child_name = item[0]
                if len(item) > 1:
                    iteration_specifier = item[-1]
                    if not isinstance(iteration_specifier, int) and \
                            iteration_specifier not in WorkflowDataTree.valid_iteration_specifier_set:
                        iteration_specifier = WorkflowDataTree.default_iterations_specifier
            else:
                child_name = item

            data = data.flatMap(lambda row: self.get_child_rows(row)) \
                .filter(lambda row: row.get(WorkflowDataTree.name_key) == child_name)

            if data.size() == 0:
                return []

            child_type = data.take(1).toList()[0].get(WorkflowDataTree.type_key)

            if child_type == WorkflowDataTree.job_type:
                return data.flatMap(lambda job: job.get(WorkflowDataTree.data_key).values()) \
                    .flatMap(lambda result_list: result_list) \
                    .toList()

            data_list = data.flatMap(lambda row: self.get_child_iteration_list(row, iteration_specifier)).toList()

            if not data_list:
                return []

        return []
        # data = stream(data_list)
        # old_size = data.size()
        # data = data.flatMap(lambda row: self.get_all_child_jobs(row))
        # size = data.size()
        # while old_size < size:
        #     data = data.flatMap(lambda row: self.get_all_child_jobs(row))
        #
        # return data.flatMap(lambda row: row.get(WorkflowDataTree.data_key).values()).toList()

    def get_path_data(self, data_path):
        root_iteration_list = [self.get_entry(WorkflowDataTree.root_key)]
        return self.get_path_data_relative(data_path, root_iteration_list)

    def get_value(self, data_path, function_list):

        data = [Result(item) for item in self.get_path_data(data_path)]

        for function_call in function_list:
            data = function_call(data)
            if isinstance(data, Result):
                data = [data]

        return data

    def get_iteration_job_path(self, iteration_row):

        iteration_row_id = iteration_row.get(WorkflowDataTree.id_key)
        if iteration_row_id == WorkflowDataTree.root_key:
            return []

        iteration_num = self.get_iteration_num(iteration_row)

        parent_row = self.get_parent_row(iteration_row)

        parent_iteration_row = self.get_parent_row(parent_row)

        iteration_job_path = self.get_iteration_job_path(parent_iteration_row)
        iteration_job_path.append([parent_row.get(WorkflowDataTree.name_key), iteration_num])

        return iteration_job_path

    def get_job_path(self, job_name, iteration_row):
        job_path = self.get_iteration_job_path(iteration_row)
        job_path.append([job_name, None])

        return job_path