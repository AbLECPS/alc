from pathlib import Path
from addict import Dict
from ConfigurationTask import ConfigurationTask
from LaunchExperimentTask import LaunchExperimentTask
from ParameterUpdates import ParameterUpdates, ParameterUpdatesAux
from WorkflowReportTask import WorkflowReportTask
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
import WorkflowParameters
from BaseJob import BaseJob
from UniqueNumberGenerator import UniqueNumberGenerator
import WorkflowUtils


class WorkflowJob(BaseJob):

    _logger = None

    _activities_key = "activities"
    _activity_name_key = "activity_name"
    _activity_node_key = "activity_node"
    _configuration_key = "configuration"
    _destination_lec_key = "destination_lec"
    _destination_list_key = "destination_list"
    _operation_key = "name"
    _parameter_updates_key = "parameter_updates"
    _source_list_key = "source_list"

    def __init__(self, parent_loop, job_name, previous_job_name_list, next_job_name_list):
        BaseJob.__init__(self, parent_loop, job_name, previous_job_name_list, next_job_name_list)

    def add_configuration(self, operation=None, source_list=None, destination_list=None, destination_lec=""):

        if source_list is None:
            source_list = []
        if destination_list is None:
            destination_list = []

        sub_map = WorkflowParameters.workflow_parameters.get_sub_map(self.path)

        if WorkflowJob._configuration_key not in sub_map:
            sub_map[WorkflowJob._configuration_key] = []

        inits_list = sub_map.get(WorkflowJob._configuration_key)

        inits_list.append({
            WorkflowJob._operation_key: operation,
            WorkflowJob._source_list_key: source_list,
            WorkflowJob._destination_list_key: destination_list,
            WorkflowJob._destination_lec_key: destination_lec
        })

        return self

    def add_activity(self, activity_name, activity_node):

        sub_map = WorkflowParameters.workflow_parameters.get_sub_map(self.path)

        if WorkflowJob._activities_key not in sub_map:
            sub_map[WorkflowJob._activities_key] = {}

        activities_map = sub_map.get(WorkflowJob._activities_key)

        activities_map[activity_name] = activity_node

        return self

    def set_parameter_updates(self, parameter_updates):
        if not isinstance(parameter_updates, ParameterUpdates) and not callable(parameter_updates):
            WorkflowJob._logger.warning(
                "ERROR: WorkflowJob.set_parameter_updates method: argument must be a ParameterUpdates.ParameterUpdates "
                "object or a function that returns a ParameterUpdates.ParameterUpdates object."
            )

        sub_map = WorkflowParameters.workflow_parameters.get_sub_map(self.path)

        sub_map[WorkflowJob._parameter_updates_key] = parameter_updates

        return self

    @staticmethod
    def get_parameter_value(parameter_map, key, default, workflow_data):
        value = parameter_map.get(key, default)
        if callable(value):
            value = value(workflow_data)
        return value

    @staticmethod
    def get_start_message(job_name):
        return "Job \"{0}\" starting ...".format(job_name)

    @staticmethod
    def get_success_message(job_name):
        return "Job \"{0}\" terminated successfully".format(job_name)

    @staticmethod
    def get_failure_message(job_name):
        return "Job \"{0}\" failed".format(job_name)

    def execute(self, state, workflow_data_tree, parameters, directory, parent_iteration_row):

        job_directory = Path(directory, self.job_name)
        self.set_runtime_id(job_directory)

        job_path = workflow_data_tree.get_job_path(self.job_name, parent_iteration_row)
        job_name = job_path[-1][0]

        workflow_report_task = WorkflowReportTask(
            job_directory,
            job_path,
            WorkflowJob.get_start_message(job_name),
            WorkflowJob.get_success_message(job_name)
        )

        failure = True
        try:
            workflow_report_task.execute()

            new_row = workflow_data_tree.add_job_data(self, None, parent_iteration_row.get(WorkflowDataTree.id_key))

    #        workflow_data_tree.save_to_file(Path(job_directory, "pre_workflow_data_tree.json"))

            sub_map = parameters.get_sub_map(self.path)

            workflow_data = WorkflowData(state, workflow_data_tree, new_row)

            config_unique_number_generator = UniqueNumberGenerator()

            for configuration_task_parameters in sub_map.get(WorkflowJob._configuration_key, []):
                ConfigurationTask(job_directory, job_path, config_unique_number_generator)\
                    .set_operation(WorkflowJob.get_parameter_value(
                        configuration_task_parameters, WorkflowJob._operation_key, [], workflow_data
                    ))\
                    .set_source_list(WorkflowJob.get_parameter_value(
                        configuration_task_parameters, WorkflowJob._source_list_key, [], workflow_data
                    ))\
                    .set_destination_list(WorkflowJob.get_parameter_value(
                        configuration_task_parameters, WorkflowJob._destination_list_key, [], workflow_data
                    ))\
                    .set_destination_lec(WorkflowJob.get_parameter_value(
                        configuration_task_parameters, WorkflowJob._destination_lec_key, "", workflow_data
                    ))\
                    .execute()

            parameter_updates = sub_map.get(WorkflowJob._parameter_updates_key, ParameterUpdates())
            if callable(parameter_updates):
                parameter_updates = parameter_updates(workflow_data)

            exec_unique_number_generator = UniqueNumberGenerator()
            output_data = Dict()
            for activity_name, activity_node in sub_map.get(WorkflowJob._activities_key, {}).items():
                activity_parameter_updates = ParameterUpdatesAux.get_activity_parameters(parameter_updates, activity_name)
                launch_experiment_task = LaunchExperimentTask(
                    job_directory,
                    job_path,
                    activity_name,
                    activity_node,
                    activity_parameter_updates,
                    exec_unique_number_generator
                ).execute()

                output_data[activity_name] = launch_experiment_task.get_data().get("Test")

            new_row[WorkflowDataTree.data_key] = output_data

            failure = False
        finally:
            if failure:
                workflow_report_task.set_failure(WorkflowJob.get_failure_message(job_name))
            workflow_report_task.execute()

#        workflow_data_tree.save_to_file(Path(job_directory, "post_workflow_data_tree.json"))

        return state, workflow_data_tree, {}


WorkflowJob._logger = WorkflowUtils.get_logger(WorkflowJob)
