from WorkflowDataTree import WorkflowDataTree


class BaseOperation:

    def __init__(self, previous_job_name_list, next_job_name_list):

        self.previous_job_name_set = set(previous_job_name_list)
        self.next_job_name_set = set(next_job_name_list)
        self.input_map = {}

    def add_input(self, input_name, job_path_list):
        self.input_map[input_name] = job_path_list
        return self

    def get_input_map(self, workflow_data_tree):

        local_input_map = {}
        for input_name, input_job_path_list in self.input_map.items():

            input_data = []
            for input_job_path in input_job_path_list:
                input_component = []
                for entry in workflow_data_tree.get_path_data(input_job_path):
                    data = entry.get(WorkflowDataTree.data_key)
                    if isinstance(data, list):
                        input_component.extend(data)
                    else:
                        input_component.append(data)
                input_data.extend(input_component)

            local_input_map[input_name] = input_data

        return local_input_map

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def get_previous_job_name_set(self):
        return set(self.previous_job_name_set)

    def get_next_job_name_set(self):
        return set(self.next_job_name_set)
