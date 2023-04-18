from pathlib import Path
from ConfigurationTask import ConfigurationTask
from LaunchExperimentTask import LaunchExperimentTask
from WorkflowReportTask import WorkflowReportTask
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
from BaseJob import BaseJob
from UniqueNumberGenerator import UniqueNumberGenerator
import WorkflowUtils


class LaunchExperimentJob(BaseJob):

    _logger = None

    _activities_key = "activities"
    _activity_name_key = "activity_name"
    _activity_node_key = "activity_node"
    _configuration_key = "configuration"
    _destination_lec_key = "destination_lec"
    _destination_list_key = "destination_list"
    _operation_key = "name"
    _source_list_key = "source_list"

    def __init__(self, job_name, previous_job_name_list, next_job_name_list):
        BaseJob.__init__(self, job_name, previous_job_name_list, next_job_name_list)

    def add_configuration(self, operation=None, source_list=None, destination_list=None, destination_lec=""):

        if source_list is None:
            source_list = []
        if destination_list is None:
            destination_list = []

        if LaunchExperimentJob._configuration_key not in self.static_parameters:
            self.static_parameters[LaunchExperimentJob._configuration_key] = []

        inits_list = self.static_parameters.get(LaunchExperimentJob._configuration_key)

        inits_list.append({
            LaunchExperimentJob._operation_key: operation,
            LaunchExperimentJob._source_list_key: source_list,
            LaunchExperimentJob._destination_list_key: destination_list,
            LaunchExperimentJob._destination_lec_key: destination_lec
        })

        return self

    def add_activity(self, activity_name, activity_node):

        if LaunchExperimentJob._activities_key not in self.static_parameters:
            self.static_parameters[LaunchExperimentJob._activities_key] = {}

        activities_map = self.static_parameters.get(LaunchExperimentJob._activities_key)

        activities_map[activity_name] = activity_node

        return self

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
            self.set_runtime_id(job_dir_path)
            new_row = workflow_data_tree.add_job_data(self, None, parent_iteration_row.get(WorkflowDataTree.id_key))

            job_path = workflow_data_tree.get_job_path(self.job_name, parent_iteration_row)

            # workflow_data_tree.save_to_file(Path(job_dir_path, "pre_workflow_data_tree.json"))

            workflow_data = WorkflowData(state, workflow_data_tree, new_row)

            config_unique_number_generator = UniqueNumberGenerator()

            for configuration_task_parameters in self.static_parameters.get(LaunchExperimentJob._configuration_key, []):
                ConfigurationTask(job_dir_path, job_path, config_unique_number_generator)\
                    .set_operation(LaunchExperimentJob.get_parameter_value(
                        configuration_task_parameters, LaunchExperimentJob._operation_key, [], workflow_data
                    ))\
                    .set_source_list(LaunchExperimentJob.get_parameter_value(
                        configuration_task_parameters, LaunchExperimentJob._source_list_key, [], workflow_data
                    ))\
                    .set_destination_list(LaunchExperimentJob.get_parameter_value(
                        configuration_task_parameters, LaunchExperimentJob._destination_list_key, [], workflow_data
                    ))\
                    .set_destination_lec(LaunchExperimentJob.get_parameter_value(
                        configuration_task_parameters, LaunchExperimentJob._destination_lec_key, "", workflow_data
                    ))\
                    .execute()

            updated_execution_parameters = self.get_updated_execution_parameters(workflow_data, execution_parameters)

            updated_execution_parameters = self.filter_parameters(updated_execution_parameters)

            exec_unique_number_generator = UniqueNumberGenerator()
            output_data = []
            for activity_name, activity_node in self.static_parameters.get(
                    LaunchExperimentJob._activities_key, {}
            ).items():
                launch_experiment_task = LaunchExperimentTask(
                    job_dir_path,
                    job_path,
                    activity_name,
                    activity_node,
                    updated_execution_parameters,
                    exec_unique_number_generator
                ).execute()

                data = launch_experiment_task.get_data()
                if isinstance(data, list):
                    output_data.extend(data)
                else:
                    output_data.append(data)

            new_row[WorkflowDataTree.data_key] = output_data

            success = True
        finally:
            workflow_report_task.execute_finish(
                success, self.get_success_message() if success else self.get_failure_message())

#        workflow_data_tree.save_to_file(Path(job_dir_path, "post_workflow_data_tree.json"))

        return state, workflow_data_tree, {}


LaunchExperimentJob._logger = WorkflowUtils.get_logger(LaunchExperimentJob)
