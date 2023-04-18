class BranchPoint:

    def __init__(self, condition_function, previous_job_name_list, next_job_name_list, branch_job_name_list):
        self.condition_function = condition_function
        self.previous_job_name_set = set(previous_job_name_list)
        self.next_job_name_set = set(next_job_name_list)
        self.branch_job_name_set = set(branch_job_name_list)

    def execute(self, workflow_data):

        actual_next_job_name_set = self.branch_job_name_set \
            if self.condition_function(workflow_data) else self.next_job_name_set

        return actual_next_job_name_set

    def get_previous_job_name_set(self):
        return set(self.previous_job_name_set)

    def get_next_job_name_set(self):
        return self.next_job_name_set.union(self.branch_job_name_set)
