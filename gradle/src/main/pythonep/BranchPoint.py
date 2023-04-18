from BaseOperation import BaseOperation
from Result import Result


class BranchPoint(BaseOperation):

    def __init__(self, condition_function, previous_job_name_list, false_next_job_name_list, true_next_job_name_list):
        BaseOperation.__init__(self, previous_job_name_list, false_next_job_name_list)
        self.condition_function = condition_function
        self.true_next_job_name_set = set(true_next_job_name_list)

    def get_next_job_name_set(self):
        return self.next_job_name_set.union(self.true_next_job_name_set)

    def execute(self, workflow_data_tree):

        local_input_map = self.get_input_map(workflow_data_tree)

        for input_name, input_data in local_input_map.items():
            local_input_map[input_name] = [Result(item) for item in input_data]

        actual_next_job_name_set = self.true_next_job_name_set \
            if self.condition_function(**local_input_map) else self.next_job_name_set

        return actual_next_job_name_set
