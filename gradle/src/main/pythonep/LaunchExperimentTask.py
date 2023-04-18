import time
import json
from pathlib import Path
from addict import Dict
from WebGMETask import WebGMETask
from ResultsTask import ResultsTask
from ProjectParameters import ProjectParameters
import WorkflowUtils
import inotify.adapters
import inotify.constants


class LaunchExperimentTask(WebGMETask):

    logger = None
    _execution_plugin_name = "LaunchExpt"
    _execution_directory_name = "execution"
    _finished = "Finished"
    _finished_with_errors = "Finished_w_Errors"

    def get_command(self):
        return [
            "node",
            "/alc/webgme/node_modules/webgme-engine/src/bin/run_plugin.js",
            LaunchExperimentTask._execution_plugin_name,
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
            "runWithJobManager": True,
            "WF_Output_Path": str(self.execution_file_path.absolute()),
            "ParamUpdates": self.parameter_updates if self.parameter_updates is not None else {}
        }

    def __init__(self, directory, job_path, activity_name, activity_node, parameter_updates, unique_number_generator):

        self.activity_name = activity_name
        self.activity_node = activity_node
        self.parameter_updates = parameter_updates

        iteration_num = 0 if len(job_path) < 2 else job_path[-2][1]

        self.timestamp = int(time.time() * 1000)
        task_name = "execute-{0}-{1}".format(self.activity_name, unique_number_generator.get_unique_number())
        self.execution_name = "{0}-{1}-iteration-{2}".format(task_name, self.timestamp, iteration_num)

        task_directory = Path(directory, task_name)
        WebGMETask.__init__(self, task_directory, task_name, job_path)

        execution_directory = Path(task_directory, LaunchExperimentTask._execution_directory_name)
        execution_directory.mkdir(parents=True, exist_ok=True)

        self.execution_file_path = Path(execution_directory, "complete")

        with self.node_task_file.open("w") as node_task_fp:
            json.dump(self._get_execute_task_file_contents(), node_task_fp, indent=2, sort_keys=True)

        self.command = self.get_command()

        self.results_task = ResultsTask(directory, self, job_path)

    def execute(self):

        self.delay_set_complete()

        # SET UP TO WAIT FOR EXECUTION FILE TO BE CREATED
        execution_file_parent_path = self.execution_file_path.parent
        execution_file_watcher = inotify.adapters.Inotify()
        file_wd = execution_file_watcher.add_watch(
            str(execution_file_parent_path), inotify.constants.IN_CREATE
        )

        WebGMETask.execute(self)

        LaunchExperimentTask.logger.info(
            "LaunchExperimentTask:  Starting wait for execution file creation \"{0}\" ...".format(
                self.execution_file_path
            )
        )

        # WAIT FOR EXECUTION FILE TO BE CREATED (WITHOUT POLLING)
        for event in execution_file_watcher.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event
            if self.execution_file_path.exists():
                break
        execution_file_watcher.remove_watch_with_id(file_wd)

        # SET UP TO WAIT UNTIL SOMETHING HAS BEEN WRITTEN TO THE EXECUTION FILE
        file_wd = execution_file_watcher.add_watch(str(self.execution_file_path), inotify.constants.IN_CLOSE_WRITE)

        # WAIT FOR SOMETHING TO BE WRITTEN TO THE EXECUTION FILE
        if self.execution_file_path.stat().st_size == 0:
            next(execution_file_watcher.event_gen(yield_nones=False))
        execution_file_watcher.remove_watch_with_id(file_wd)

        LaunchExperimentTask.logger.info(
            "LaunchExperimentTask:  Execution file detected! \"{0}\" ...".format(
                self.execution_file_path
            )
        )

        with self.execution_file_path.open("r") as execution_file_fp:
            completion_string = execution_file_fp.read().strip()

        if completion_string == LaunchExperimentTask._finished_with_errors:
            self.write_to_failing_task_path_file()
            raise Exception(
                "Task \"{0}\" of job \"{1}\" has failed execution, i.e. \"{2}\" was written to completion file "
                "\"{3}\"".format(
                    self.task_name, self.job_path, LaunchExperimentTask._finished_with_errors, self.execution_file_path
                )
            )

        if completion_string != LaunchExperimentTask._finished:
            self.write_to_failing_task_path_file()
            raise Exception(
                "Task \"{0}\" of job \"{1}\" has failed execution, unexpected string \"{2}\" was written to "
                "completion file \"{3}\"".format(
                    self.task_name, self.job_path, completion_string, self.execution_file_path
                )
            )

        self.set_complete()

        self.results_task.execute()

        return self

    def get_data(self):
        with self.results_task.output_file.open() as results_fp:
            return json.load(results_fp, object_hook=lambda x: Dict(x))


LaunchExperimentTask.logger = WorkflowUtils.get_logger(LaunchExperimentTask)