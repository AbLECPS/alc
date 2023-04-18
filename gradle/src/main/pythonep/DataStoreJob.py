from pathlib import Path
import re
import json

import ConfigKeys
from BaseJob import BaseJob
from WorkflowReportTask import WorkflowReportTask
from WorkflowDataTree import WorkflowDataTree
from ProjectParameters import ProjectParameters


class DataStoreJob(BaseJob):

    directory_re = re.compile("[0-9]+")

    all_key = "all"
    latest_key = "latest"
    unused_key = "unused"

    def __init__(self, job_name, next_job_name_list, json_content):
        BaseJob.__init__(self, job_name, [], next_job_name_list)

        self.job_name = job_name
        self.json_content = json_content

    def get_start_message(self):
        return "DataStoreJob \"{0}\" placing data in workflow ...".format(self.job_name)

    def get_success_message(self):
        return "DataStoreJob \"{0}\" terminated successfully".format(self.job_name)

    @staticmethod
    def get_current_workflow_data_node_set():

        current_workflow_data_node_set = set()

        execution_dir_path = Path(ProjectParameters.get_execution_dir_path())

        workflow_executions_path = execution_dir_path.parent

        child_directory_list = [
            x for x in workflow_executions_path.iterdir() if x.is_dir() or DataStoreJob.directory_re.fullmatch(x.name)
        ]

        if len(child_directory_list) == 0:
            return {}

        # GET ALL DATA NODES FROM THIS WORKFLOW
        for child_directory in child_directory_list:

            workflow_data_tree_path = Path(child_directory, "data", "workflow_data_tree.json")

            if not workflow_data_tree_path.exists():
                continue

            with workflow_data_tree_path.open("r") as json_fp:
                workflow_json = json.load(json_fp)

            if workflow_json[WorkflowDataTree.root_key][WorkflowDataTree.exit_status_key] != 0:
                continue

            for value in workflow_json.values():
                if len(value.get(WorkflowDataTree.child_id_list_key)) == 0:

                    data_list = value.get(WorkflowDataTree.data_key, [])
                    for data in data_list:
                        if WorkflowDataTree.path_key in data:
                            current_workflow_data_node_set.add(data.get(WorkflowDataTree.path_key))

        return current_workflow_data_node_set

    def get_job_data_dict(self, job_data):

        flag = job_data.get(ConfigKeys.key_key, DataStoreJob.latest_key)
        workflow_directory_name = job_data.get(ConfigKeys.workflow_key, None)
        path_list = job_data.get(ConfigKeys.job_data_key, None)

        if workflow_directory_name is None or path_list is None:
            return {}

        data_node_path_dict = {}

        execution_dir_path = Path(ProjectParameters.get_execution_dir_path())

        workflow_executions_path = Path(execution_dir_path.parent.parent, workflow_directory_name)

        child_directory_list = [
            x for x in workflow_executions_path.iterdir() if x.is_dir() or DataStoreJob.directory_re.fullmatch(x.name)
        ]

        if len(child_directory_list) == 0:
            return {}

        if flag == DataStoreJob.latest_key:
            timestamp_list = [int(x.name) for x in child_directory_list]
            timestamp_list.sort()
            latest_timestamp = timestamp_list[-1]
            child_directory_list = [Path(workflow_executions_path, str(latest_timestamp))]

        # GET ALL DATA NODES FROM THIS WORKFLOW
        for child_directory in child_directory_list:

            workflow_data_tree_path = Path(child_directory, "data", "workflow_data_tree.json")

            if not workflow_data_tree_path.exists():
                continue

            workflow_data_tree = WorkflowDataTree()
            workflow_data_tree.load_from_file(workflow_data_tree_path)

            root_node_data = workflow_data_tree.get_entry(WorkflowDataTree.root_key)
            if root_node_data[WorkflowDataTree.exit_status_key] != 0:
                continue

            data_list = []
            for node in workflow_data_tree.get_path_data(path_list):
                data = node[ConfigKeys.json_data_key]
                if isinstance(data, list):
                    data_list.extend(data)
                else:
                    data_list.append(data)

            if flag == DataStoreJob.unused_key:
                data_list = filter(
                    lambda x: WorkflowDataTree.path_key in x and
                              x[WorkflowDataTree.path_key] not in self.get_current_workflow_data_node_set(),
                    data_list
                )

            data_node_path_dict.update(
                {x[WorkflowDataTree.path_key]: x for x in data_list if WorkflowDataTree.path_key in x}
            )

        return data_node_path_dict

    def execute(self, state, workflow_data_tree, parameters, directory, parent_iteration_row):

        workflow_report_task = WorkflowReportTask()
        workflow_report_task.execute_start(self.get_start_message())

        job_dir_path = Path(directory, self.job_name)
        self.set_runtime_id(job_dir_path)

        job_data = []
        if ConfigKeys.json_data_key in self.json_content:
            job_data.extend(self.json_content[ConfigKeys.json_data_key])

        if ConfigKeys.job_data_key in self.json_content:
            job_data_dict = self.get_job_data_dict(self.json_content[ConfigKeys.job_data_key])
            job_data.extend(job_data_dict.values())

        workflow_data_tree.add_job_data(
            self, job_data, parent_iteration_row.get(WorkflowDataTree.id_key)
        )

        workflow_report_task.execute_finish(True, self.get_success_message())

        return state, workflow_data_tree, {}
