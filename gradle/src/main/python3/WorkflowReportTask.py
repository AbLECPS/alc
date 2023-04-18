from datetime import datetime
from ProjectParameters import ProjectParameters
from WebGMETask import WebGMETask


class WorkflowReportTask:

    workflow_path = [["workflow", None]]

    def __init__(self, directory, job_path, start_message, finish_message):
        self.job_path = job_path
        self.success = None
        self.finish_success = True
        self.message = start_message
        self.finish_message = finish_message
        self.extra_data = {}
        self.start_time = None
        self.finish_time = None

        self.start_inspect_task = None
        self.finish_inspect_task = None
        self.inspect_task = None
        if ProjectParameters.has_status_node():
            self.task_name = "start"
            self.start_inspect_task = WebGMETask.get_inspect_task(directory, self.job_path, self)
            self.task_name = "termination"
            self.finish_inspect_task = WebGMETask.get_inspect_task(directory, self.job_path, self)
            self.task_name = None
            self.inspect_task = self.start_inspect_task

    @staticmethod
    def get_current_time():
        return "{0}Z".format(datetime.now().isoformat())

    def set_extra_data(self, extra_data):
        self.extra_data = extra_data

    def get_status_map(self):
        status_map = {
            "success":    self.success,
            "messages":   [self.message],
            "artifacts":  [],
            "pluginName": "",
            "startTime":  self.start_time,
            "finishTime": self.finish_time,
            "error":      None,
            "projectId":  ProjectParameters.get_project_id(),
            "pluginId":   "",
            "commits":    []
        }

        status_map.update(self.extra_data)

        return status_map

    def get_was_complete(self):
        return False

    def set_failure(self, finish_message):
        self.finish_success = False
        self.finish_message = finish_message

    def execute(self):
        if self.start_time is None:
            self.start_time = WorkflowReportTask.get_current_time()
        else:
            self.finish_time = WorkflowReportTask.get_current_time()
            self.success = self.finish_success
            self.message = self.finish_message
            self.inspect_task = self.finish_inspect_task

        if self.inspect_task is not None:
            self.inspect_task.execute()

        self.extra_data = {}
        return self
