import json
import traceback
from ResultsTask import ResultsTask
from UpdateStatusTask import UpdateStatusTask
from WorkflowDataTree import WorkflowDataTree
import WorkflowUtils


class InspectTask(UpdateStatusTask):

    _error_key = "error"
    _job_path_key = "job_path"
    _messages_key = "messages"
    _task_name_key = "task_name"
    _results_key = "results"

    _skipped_message = "SKIPPED"

    def __init__(self, directory, inspected_task, job_path, status_path):
        self.inspected_task = inspected_task
        self.status_path = status_path
        self.job_path = job_path
        self.base_directory = directory
        self.task_name = "inspect-{0}".format(self.inspected_task.task_name)

    def execute(self):

        try:
            status_map = self.inspected_task.get_status_map()

            status_map[InspectTask._job_path_key] = self.job_path
            status_map[InspectTask._task_name_key] = self.inspected_task.task_name

            if self.inspected_task.get_was_complete():
                if InspectTask._messages_key not in status_map:
                    status_map[InspectTask._messages_key] = []
                status_map.get(InspectTask._messages_key).append(InspectTask._skipped_message)

            if isinstance(self.inspected_task, ResultsTask):
                with self.inspected_task.output_file.open("r") as results_fp:
                    results_output = json.load(results_fp)

                status_map[InspectTask._results_key] = [{
                    WorkflowDataTree.path_key:
                        result.get(WorkflowDataTree.path_key),
                    WorkflowDataTree.result_url_key:
                        result.get(WorkflowDataTree.result_url_key),
                    WorkflowDataTree.directory_key:
                        result.get(WorkflowDataTree.directory_key)
                } for result in results_output]

            message = json.dumps(status_map, sort_keys=True)

            with WorkflowUtils.lock:
                self.update_status_node(message)

            if status_map[InspectTask._error_key] is not None:
                raise Exception(status_map.get(InspectTask._error_key, ""))

            return self

        except Exception as e:
            message = "WARNING: inspect_task failed for task \"{0}\":  Exception message: {1}, check files in " \
                      "directory \"{2}\" for more information\nStack Trace:\n{3}" \
                .format(self.task_name, e, self.base_directory.absolute(), traceback.format_exc())
            InspectTask.logger.warning(message)


InspectTask.logger = WorkflowUtils.get_logger(InspectTask)
