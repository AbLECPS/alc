from BaseJob import BaseJob
from WorkflowDataTree import WorkflowDataTree
from Result import Result
from pathlib import Path


class Transform(BaseJob):

    def __init__(self, transform_function, job_name, previous_job_name_list, next_job_name_list):
        BaseJob.__init__(self, job_name, previous_job_name_list, next_job_name_list)

        self.transform_function = transform_function

    def execute(self, state, workflow_data_tree, execution_parameters, directory, parent_iteration_row):

        job_dir_path = Path(directory, self.job_name)
        self.set_runtime_id(job_dir_path)
        new_row = workflow_data_tree.add_job_data(self, None, parent_iteration_row.get(WorkflowDataTree.id_key))

        job_dir_path.mkdir(parents=True, exist_ok=True)

        local_input_map = self.get_input_map(workflow_data_tree)

        for input_name, input_data in local_input_map.items():
            local_input_map[input_name] = [Result(item) for item in input_data]

        output_data = self.transform_function(**local_input_map)
        new_row[WorkflowDataTree.data_key] = output_data

        return state, workflow_data_tree, {}
