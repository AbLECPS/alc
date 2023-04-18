class WorkflowParameters:

    def __init__(self):
        self.job_parameters = {}

    def get_sub_map(self, workflow_path):
        key_list = workflow_path.split('.')
        current_dict = self.job_parameters
        for key in key_list:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict.get(key)

        return current_dict


# FOR COMPILE-TIME INITIALIZATION
workflow_parameters = WorkflowParameters()
