from datetime import datetime
from ProjectParameters import ProjectParameters
import json
import WorkflowUtils
from UpdateStatusTask import UpdateStatusTask


class WorkflowReportTask(UpdateStatusTask):

    def __init__(self):
        self.start_time = None

    @staticmethod
    def get_current_time():
        return "{0}Z".format(datetime.now().isoformat())

    @staticmethod
    def get_status_map(success, message, extra_data, start_time, finish_time):
        status_map = {
            "success": success,
            "messages": [message],
            "artifacts": [],
            "pluginName": "",
            "startTime": start_time,
            "finishTime": finish_time,
            "error": None,
            "projectId": ProjectParameters.get_project_id(),
            "pluginId": "",
            "commits": []
        }

        status_map.update(extra_data)

        return status_map

    def execute_start(self, message, extra_data=None):

        if extra_data is None:
            extra_data = {}

        if ProjectParameters.has_status_node():
            self.start_time = self.get_current_time()
            update_message = json.dumps(self.get_status_map(None, message, extra_data, self.start_time, None))
            self.update_status_node(update_message)
        else:
            self.logger.warning("Unable to update status node with start message:  No status node specified")

    def execute_finish(self, success, message, extra_data=None):

        if extra_data is None:
            extra_data = {}

        if ProjectParameters.has_status_node():
            update_message = json.dumps(self.get_status_map(
                success, message, extra_data, self.start_time, self.get_current_time()
            ))
            self.update_status_node(update_message)
        else:
            self.logger.warning("Unable to update status node with finish message:  No status node specified")


WorkflowReportTask.logger = WorkflowUtils.get_logger(WorkflowReportTask)
