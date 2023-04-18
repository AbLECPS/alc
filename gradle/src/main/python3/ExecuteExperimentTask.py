import time
import json
from pathlib import Path
from addict import Dict
from WebGMETask import WebGMETask
from ResultsTask import ResultsTask
from ProjectParameters import ProjectParameters
import WorkflowUtils


class ExecuteExperimentTask(WebGMETask):

    _execution_plugin_name = "ExecuteExpt"

    def get_command(self):
        return [
            "node",
            "/alc/webgme/node_modules/webgme-engine/src/bin/run_plugin.js",
            ExecuteExperimentTask._execution_plugin_name,
            ProjectParameters.get_project_name(),
            "-a", self.activity_node,
            "-u", WorkflowUtils.user_arg,
            "-o", ProjectParameters.get_owner(),
            "-l", WorkflowUtils.webgme_url,
            "-j", str(self.node_task_file.absolute()),
            "-n", ProjectParameters.get_namespace()
        ]

    def _get_execute_task_file_contents(self):
        return {
            "name": self.execution_name,
            "setupJupyterNB": False,
            "generateROSLaunch": False,
            "ParamUpdates": self.parameter_updates if self.parameter_updates is not None else {}
        }

    def __init__(self, directory, job_path, activity_name, activity_node, parameter_updates, unique_number_generator):

        self.activity_name = activity_name
        self.activity_node = activity_node
        self.parameter_updates = parameter_updates

        self.timestamp = int(time.time() * 1000)
        task_name = "execute-{0}-{1}".format(self.activity_name, unique_number_generator.get_unique_number())
        self.execution_name = "{0}-{1}".format(task_name, self.timestamp)

        execute_directory = Path(directory, task_name)
        WebGMETask.__init__(self, execute_directory, task_name, job_path)

        with self.node_task_file.open("w") as node_task_fp:
            json.dump(self._get_execute_task_file_contents(), node_task_fp, indent=2, sort_keys=True)

        self.command = self.get_command()

        self.results_task = ResultsTask(directory, self, job_path)

    def execute(self):

        WebGMETask.execute(self)

        self.results_task.execute()

        return self

    def get_data(self):
        with self.results_task.output_file.open() as results_fp:
            return json.load(results_fp, object_hook=lambda x: Dict(x))
