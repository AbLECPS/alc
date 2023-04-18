from pathlib import Path
from addict import Dict
import json
import shutil
import os
from SlurmTask import SlurmTask
from WorkflowReportTask import WorkflowReportTask
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
from BaseJob import BaseJob
from ProjectParameters import ProjectParameters
import WorkflowUtils


class CompletionIndicator:

    _complete_job_file_name = "__COMPLETE__"

    def __init__(self, directory):
        self.complete_file_path = Path(directory, self._complete_job_file_name)

    def test_complete(self):
        return self.complete_file_path.exists()

    def set_complete(self):
        self.complete_file_path.touch()


class LaunchActivityJob(BaseJob):

    _logger = None

    _attributes_key = "attributes"
    _node_named_path_key = "node_named_path"
    _node_path_key = "node_path"

    _execution_file_name = "run.sh"
    _symlink_file_name_list = ["main.py", _execution_file_name]
    _input_file_name = "input.json"
    _inputs_dir_name = "Inputs"
    _launch_activity_file_name = "launch_activity_output.json"
    _parameters_file_name = "parameters.json"
    _prototype_dir_name = "Prototype"
    _standard_output_file_name = "slurm_job_log.txt"

    def __init__(self, job_name, activity_name, activity_node, previous_job_name_list, next_job_name_list):
        BaseJob.__init__(self, job_name, previous_job_name_list, next_job_name_list)
        self.activity_name = activity_name
        self.activity_node = activity_node

    @staticmethod
    def get_parameter_value(parameter_map, key, default, workflow_data):
        value = parameter_map.get(key, default)
        if callable(value):
            value = value(workflow_data)
        return value

    def get_start_message(self):
        return "Job \"{0}\" starting ...".format(self.job_name)

    def get_success_message(self):
        return "Job \"{0}\" terminated successfully".format(self.job_name)

    def get_failure_message(self):
        return "Job \"{0}\" failed".format(self.job_name)

    def execute(self, state, workflow_data_tree, execution_parameters, directory, parent_iteration_row):

        workflow_report_task = WorkflowReportTask()

        success = False
        try:
            workflow_report_task.execute_start(self.get_start_message())

            job_dir_path = Path(directory, self.job_name)
            completion_indicator = CompletionIndicator(job_dir_path)
            self.set_runtime_id(job_dir_path)
            new_row = workflow_data_tree.add_job_data(self, None, parent_iteration_row.get(WorkflowDataTree.id_key))
            output_file_path = Path(job_dir_path, SlurmTask.output_file_name)

            slurm_task = None
            if not completion_indicator.test_complete():
                job_dir_path.mkdir(parents=True, exist_ok=True)

                inputs_dir_path = Path(job_dir_path, LaunchActivityJob._inputs_dir_name)
                inputs_dir_path.mkdir(parents=True, exist_ok=True)

                local_input_map = self.get_input_map(workflow_data_tree)

                for input_name, input_data in local_input_map.items():
                    input_dir_path = Path(inputs_dir_path, input_name)
                    input_dir_path.mkdir(parents=True, exist_ok=True)
                    input_file_path = Path(input_dir_path, LaunchActivityJob._input_file_name)
                    with input_file_path.open("w") as input_file_fp:
                        json.dump(input_data, input_file_fp, indent=4, sort_keys=True)

                execution_dir_path = ProjectParameters.get_execution_dir_path()
                prototype_dir_path = Path(execution_dir_path, LaunchActivityJob._prototype_dir_name, self.activity_name)

                launch_activity_file_path = Path(prototype_dir_path, LaunchActivityJob._launch_activity_file_name)

                with launch_activity_file_path.open("r") as launch_activity_fp:
                    launch_activity_data = json.load(launch_activity_fp)

                launch_attributes = launch_activity_data[self._attributes_key]
                active_node = launch_attributes[self._node_path_key]

                job_launch_activity_file_path = Path(job_dir_path, LaunchActivityJob._launch_activity_file_name)
                shutil.copy(launch_activity_file_path, job_launch_activity_file_path)

                node_named_path = Path(launch_attributes[self._node_named_path_key])
                execution_name = "result-{0}".format(node_named_path.name)

                for file_name in LaunchActivityJob._symlink_file_name_list:
                    execution_file_path = Path(prototype_dir_path, file_name)
                    job_execution_file_path = Path(job_dir_path, file_name)
                    if job_execution_file_path.exists() and not job_execution_file_path.is_symlink():
                        job_execution_file_path.unlink()
                    if not job_execution_file_path.exists():
                        os.symlink(execution_file_path, job_execution_file_path)

                parameters_file_path = Path(prototype_dir_path, LaunchActivityJob._parameters_file_name)
                job_parameters_file_path = Path(job_dir_path, LaunchActivityJob._parameters_file_name)

                standard_output_file_path = Path(job_dir_path, LaunchActivityJob._standard_output_file_name)

                workflow_data = WorkflowData(state, workflow_data_tree, new_row)

                updated_execution_parameters = Dict(execution_parameters)
                if parameters_file_path.exists():
                    with parameters_file_path.open("r") as parameters_json_fp:
                        parameter_map = json.load(parameters_json_fp)
                        updated_execution_parameters.update(parameter_map)

                updated_execution_parameters = self.get_updated_execution_parameters(
                    workflow_data, updated_execution_parameters
                )

                updated_execution_parameters = self.filter_parameters(updated_execution_parameters)

                with job_parameters_file_path.open("w") as parameters_json_fp:
                    json.dump(updated_execution_parameters, parameters_json_fp, indent=4, sort_keys=True)

                # workflow_data_tree.save_to_file(Path(job_dir_path, "pre_workflow_data_tree.json"))

                slurm_task = SlurmTask(
                    active_node=active_node,
                    execution_name=execution_name,
                    working_dir=job_dir_path,
                    command=[LaunchActivityJob._execution_file_name],
                    standard_output=standard_output_file_path
                )

                slurm_task.execute()

                completion_indicator.set_complete()

            if slurm_task is None or slurm_task.get_success():
                with output_file_path.open("r") as job_data_fp:
                    job_data = json.load(job_data_fp)
            else:
                raise Exception(
                    "Workflow job \"{0}\" failed.  Check \"{1}\" directory for more information.".format(
                        self.job_name, job_dir_path
                    )
                )

            # CURRENTLY ENCOUNTERING PERMISSIONS PROBLEMS WITH THIS
            # with output_file_path.open("w") as job_data_fp:
            #     json.dump(job_data, job_data_fp, indent=4, sort_keys=True)

            new_row[WorkflowDataTree.data_key] = Dict(job_data)

            success = True
        finally:
            workflow_report_task.execute_finish(
                success, self.get_success_message() if success else self.get_failure_message()
            )

        # workflow_data_tree.save_to_file(Path(job_dir_path, "post_workflow_data_tree.json"))

        return state, workflow_data_tree, {}


LaunchActivityJob._logger = WorkflowUtils.get_logger(LaunchActivityJob)
