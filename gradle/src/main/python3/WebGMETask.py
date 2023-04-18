import os
import re
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from ExecTask import ExecTask
from ProjectParameters import ProjectParameters
import logging
import WorkflowUtils


class WebGMETask(ExecTask):

    logger = None

    _block_size = 1024
    _complete_file_name = "__COMPLETE__"

    _error_key = "error"
    _web_gme_home_dir = Path("/alc/webgme")

    _unique_id = 0

    _max_retry = 100
    _retry_sleep_interval = 3

    _lock = threading.Lock()

    _failure_text = "\"success\": false"
    _connection_error_text = "\"error\": \"Address already in use\""

    _text_set = {_failure_text, _connection_error_text}

    @staticmethod
    def get_unique_id():
        retval = WebGMETask._unique_id
        WebGMETask._unique_id += 1
        return retval

    @staticmethod
    def _get_last_right_brace_pos(fp):
        fp.seek(0, os.SEEK_END)
        old_pos = fp.tell()
        pos = old_pos - WebGMETask._block_size
        bytes_to_read = WebGMETask._block_size
        if pos < 0:
            pos = 0
            bytes_to_read = old_pos

        last_brace_pos = -1
        while bytes_to_read > 0 and last_brace_pos < 0:

            fp.seek(pos, os.SEEK_SET)
            read_string = fp.read(bytes_to_read)

            last_brace_pos = read_string.rfind("}")

            old_pos = pos
            pos -= bytes_to_read
            if pos < 0:
                pos = 0
                bytes_to_read = old_pos

        return last_brace_pos if last_brace_pos < 0 else old_pos + last_brace_pos

    @staticmethod
    def x_out_quoted_text(data):
        data = re.sub("\\\\.", "XX", data)[::-1]

        for match in re.finditer("\"[^\"]*\"", data[:-1]):
            data = data[:match.start()] + "X" * (match.end() - match.start()) + data[match.end():]

        return data[::-1]

    @staticmethod
    def get_new_value(value, search_string):
        new_left_search_pos = search_string.rfind("{")
        old_left_search_pos = new_left_search_pos

        new_right_search_pos = search_string.rfind("}")

        while value > 0 and (new_left_search_pos > 0 or new_right_search_pos > 0):
            old_left_search_pos = new_left_search_pos
            old_right_search_pos = new_right_search_pos

            if old_right_search_pos > old_left_search_pos:
                value += 1
                new_right_search_pos = search_string.rfind("}", 0, old_right_search_pos)
            else:
                value -= 1
                new_left_search_pos = search_string.rfind("{", 0, old_left_search_pos)

        return [value, old_left_search_pos]

    @staticmethod
    def get_json(path):
        if not isinstance(path, Path):
            path = Path(path)

        file_size = path.lstat().st_size
        if file_size < 2:
            return ""

        with path.open("r") as open_fp:

            # old_pos = WebGMETask._get_last_right_brace_pos(open_fp)
            # if old_pos != file_size - 2:
            #     return ""

            old_pos = file_size - 2
            open_fp.seek(old_pos)
            last_right_brace = open_fp.read(1)
            if last_right_brace != "}":
                return ""

            pos = old_pos - WebGMETask._block_size
            bytes_to_read = WebGMETask._block_size
            if pos < 0:
                pos = 0
                bytes_to_read = old_pos

            examine_string = ""
            actual_string = "}"

            value = 1
            left_search_pos = 0
            while bytes_to_read > 0 and value > 0:

                open_fp.seek(pos)
                read_string = open_fp.read(bytes_to_read)

                actual_string = read_string + actual_string
                examine_string = read_string + examine_string

                examine_string = WebGMETask.x_out_quoted_text(examine_string)

                after_last_quote_pos = examine_string.rfind("\"") + 1

                search_string = examine_string[after_last_quote_pos:]
                examine_string = examine_string[0:after_last_quote_pos]

                pair = WebGMETask.get_new_value(value, search_string)
                value = pair[0]
                left_search_pos = pair[1] + after_last_quote_pos

                old_pos = pos
                pos -= WebGMETask._block_size
                if pos < 0:
                    pos = 0
                    bytes_to_read = old_pos

            if value < 0:
                return ""
            # print("left_search_pos = {0}".format(left_search_pos))
            # print("actual_string = {0}".format(actual_string[left_search_pos:]))
            return actual_string[left_search_pos:]

    def write_status_file(self):

        status_json_string = WebGMETask.get_json(self.std_output_file)
        if len(status_json_string) == 0:
            status_json_string = WebGMETask.get_json(self.std_error_file)
            if len(status_json_string) == 0:
                raise Exception("Unable to get status of task \"{0}\"".format(self.task_name))

        status_json = json.loads(status_json_string)
        with self.status_output_file.open("w") as status_fp:
            json.dump(status_json, status_fp, indent=4, sort_keys=True)

        error = status_json.get(WebGMETask._error_key)
        if error is not None:
            raise Exception("Error occured in task \"{0}\"\n{1}".format(self.task_name, error))

    def get_default_status_map(self):
        current_time = "{0}Z".format(datetime.now().isoformat())
        return {
            "success":    False,
            "messages":   ["No status file generated for task \"{0}\"".format(self.task_name)],
            "artifacts":  [],
            "pluginName": "",
            "startTime":  current_time,
            "finishTime": current_time,
            "error":      "ERROR",
            "projectId":  ProjectParameters.get_project_id(),
            "pluginId":   "",
            "commits":    []
        }

    def get_status_map(self):

        if self.status_output_file.exists():
            with self.status_output_file.open("r") as status_file_fp:
                status_map = json.load(status_file_fp)
        else:
            status_map = self.get_default_status_map()

        return status_map

    @staticmethod
    def check_file_for_connection_error(file_fp):
        search_set = set(WebGMETask._text_set)

        line = file_fp.readline()
        while line and search_set:
            new_search_set = set()
            for text in search_set:
                if text not in line:
                    new_search_set.add(text)
            search_set = new_search_set
            line = file_fp.readline()

        return not search_set

    def check_for_connection_error(self):
        with self.std_error_file.open("r") as stderr_fp:
            if self.check_file_for_connection_error(stderr_fp):
                return True

        with self.std_output_file.open("r") as stdout_fp:
            return self.check_file_for_connection_error(stdout_fp)

    def is_complete(self):
        return self.complete_file.exists()

    def delay_set_complete(self):
        self.delay_set_complete_flag = True

    def set_complete(self):
        if self.delay_set_complete_flag:
            self.delay_set_complete_flag = False
        else:
            self.complete_file.touch()

    def set_not_complete(self):
        if self.complete_file.exists():
            self.complete_file.unlink()

    def set_was_complete(self):
        self.was_complete = self.is_complete()

    def get_was_complete(self):
        return self.was_complete

    def __init__(self, directory, task_name, job_path, enable_inspect_task=True):
        ExecTask.__init__(self)

        self.task_name = task_name
        self.job_path = job_path

        self.base_directory = directory

        self.input_dir = Path(directory, "input")
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(directory, "output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.node_task_file = Path(self.input_dir, "task.json")
        self.status_output_file = Path(self.output_dir, "status.json")
        self.std_output_file = Path(self.output_dir, "stdout")
        self.std_error_file = Path(self.output_dir, "stderr")

        self.set_working_dir(WebGMETask._web_gme_home_dir)

        self.set_standard_output(self.std_output_file)
        self.set_standard_error(self.std_error_file)

        self.was_complete = None
        self.complete_file = Path(self.base_directory, WebGMETask._complete_file_name)
        self.delay_set_complete_flag = False

        self.inspect_task = None
        if ProjectParameters.has_status_node() and enable_inspect_task:
            self.inspect_task = WebGMETask.get_inspect_task(directory.parent, job_path, self)

    def write_to_failing_task_path_file(self):
        with ProjectParameters.get_failing_task_path_file_path().open("a") as failing_task_path_fp:
            print("{0}\n".format(self.base_directory), file=failing_task_path_fp)

    def execute(self):

        self.set_was_complete()

        try:
            if not self.is_complete():

                WebGMETask.logger.info("WebGMETask:  Task \"{0}\" of Job \"{1}\" trying to acquire lock ...".format(
                    self.task_name, self.job_path
                ))
                with WebGMETask._lock:
                    WebGMETask.logger.info("WebGMETask:  Task \"{0}\" of Job \"{1}\" has acquired lock.".format(
                        self.task_name, self.job_path
                    ))

                    retry = True
                    retry_count = 0
                    while retry:
                        retry = False

                        try:
                            ExecTask.execute(self)
                        except:
                            WebGMETask.logger.info(
                                "WebGMETask:  Task \"{0}\" of Job \"{1}\" caught an exception.  "
                                "Checking if connection error.".format(self.task_name, self.job_path)
                            )

                        if self.get_exit_status() != 0:
                            if self.check_for_connection_error():
                                if retry_count < WebGMETask._max_retry:
                                    retry = True
                                    retry_count += 1
                                    WebGMETask.logger.info(
                                        "WebGMETask:  Task \"{0}\" of Job \"{1}\" has detected a connection error, "
                                        "retry_count is now {2}.  Sleeping for {3} seconds ...".format(
                                            self.task_name, self.job_path, retry_count, WebGMETask._retry_sleep_interval
                                        )
                                    )
                                    time.sleep(WebGMETask._retry_sleep_interval)
                                else:
                                    WebGMETask.logger.info(
                                        "WebGMETask:  Task \"{0}\" of Job \"{1}\" has detected a connection error, "
                                        " and retry_count ({2}) exceeded.  Raising exception.".format(
                                            self.task_name, self.job_path, WebGMETask._max_retry
                                        )
                                    )

                            if not retry:
                                self.write_to_failing_task_path_file()
                                raise Exception(
                                    "Task \"{0}\" of job \"{1}\" has exited with a non-zero exit status ({2}): "
                                    "please check files in directory \"{3}\" to determine cause of failure".format(
                                        self.task_name, self.job_path, self.get_exit_status(), self.base_directory
                                    )
                                )

                WebGMETask.logger.info(
                    "WebGMETask:  Task \"{0}\" of Job \"{1}\" has released lock.".format(self.task_name, self.job_path)
                )

                try:
                    self.write_status_file()
                except Exception as e:
                    self.write_to_failing_task_path_file()
                    raise Exception(
                        "Unable to get status file of task \"{0}\" of job \"{1}\": "
                        "Exception message: \"{2}\", "
                        "please check files in directory \"{3}\" to determine cause of failure".format(
                            self.task_name, self.job_path, e, self.base_directory
                        )
                    )

                self.set_complete()
        finally:
            if self.inspect_task is not None:
                self.inspect_task.execute()

        return self

    @staticmethod
    def get_inspect_task(directory, job_path, inspected_task):
        from InspectTask import InspectTask
        return InspectTask(directory, inspected_task, job_path, ProjectParameters.get_status_node())


WebGMETask.logger = WorkflowUtils.get_logger(WebGMETask)
WebGMETask.logger.setLevel(logging.INFO)
