from pathlib import Path
import json
from WebGMETask import WebGMETask
from ProjectParameters import ProjectParameters
import WorkflowUtils


class ALCModelUpdaterTask(WebGMETask):

    _alc_plugin_name = "ALCModelUpdater"

    def get_command(self):
        return [
            "node",
            "/alc/webgme/node_modules/webgme-engine/src/bin/run_plugin.js",
            ALCModelUpdaterTask._alc_plugin_name,
            ProjectParameters.get_project_name(),
            "-a", ProjectParameters.get_generic_active_node(),
            "-u", WorkflowUtils.user_arg,
            "-o", ProjectParameters.get_owner(),
            "-l", WorkflowUtils.webgme_url,
            "-j", str(self.node_task_file.absolute()),
            "-n", ProjectParameters.get_namespace()
        ]

    @staticmethod
    def write_node_task_file(task_file_fp, input_file_path):
        json_data = {"task": str(input_file_path.absolute())}
        json.dump(json_data, task_file_fp, indent=4, sort_keys=True)

    def __init__(self, directory, task_name, job_path, enable_inspect_task=True):
        WebGMETask.__init__(self, directory, task_name, job_path, enable_inspect_task=enable_inspect_task)

        self.directory = directory

        self.input_file = Path(self.input_dir, "input.json")
        self.output_file = Path(self.output_dir, "output.json")

        self.command = self.get_command()

        with self.node_task_file.open("w") as node_task_fp:
            ALCModelUpdaterTask.write_node_task_file(node_task_fp, self.input_file)
