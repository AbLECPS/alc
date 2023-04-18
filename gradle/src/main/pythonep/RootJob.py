from pathlib import Path
from addict import Dict
from DataStoreJob import DataStoreJob
from SequenceJob import SequenceJob
from WorkflowDataTree import WorkflowDataTree
from WorkflowReportTask import WorkflowReportTask
from ProjectParameters import ProjectParameters
import ScriptFileNames
import WorkflowUtils
import DiagnosticKeys


class RootJob(SequenceJob):

    logger = None

    _workflow_start_message = "workflow starting ..."

    _workflow_success_message = "Workflow complete"
    _workflow_failure_message = "Workflow failed"

    _workflow_path = "workflow"

    _report_task_dir_name = "run"

    def __init__(self, workflow_name, workflow_dir):
        SequenceJob.__init__(self, "workflow", [], [])
        self.workflow_dir = workflow_dir
        self.set_is_root()

    def add_data_store_job(self, job_name, next_job_name_list, json_content):
        new_job = DataStoreJob(job_name, next_job_name_list, json_content)
        self.job_map[job_name] = new_job

    @staticmethod
    def print_exception_map(exception_map, prefix_list, exception_fp):
        for sub_job_name, job_exception_info in exception_map.items():
            new_prefix_list = prefix_list + [sub_job_name]
            message_prefix = " -> ".join(new_prefix_list)
            if isinstance(job_exception_info, tuple):
                exception, stack_trace = job_exception_info
                message = "{0}:  Message: {1}:  \nStackTrace:\n{2}".format(message_prefix, exception, stack_trace)
#                RootJob.logger.error(message)
                print(message, file=exception_fp)
            else:
                RootJob.print_exception_map(job_exception_info, new_prefix_list, exception_fp)

    def execute_workflow(self):

        job_data = Path(self.workflow_dir, ScriptFileNames.job_data_dir_name)
        job_data.mkdir(parents=True, exist_ok=True)

        workflow_data_tree_dir = Path(self.workflow_dir, ScriptFileNames.data_dir_name)
        workflow_data_tree_dir.mkdir(parents=True, exist_ok=True)

        workflow_data_tree_json = Path(workflow_data_tree_dir, ScriptFileNames.workflow_data_tree_file_name)

        exceptions_file_path = ProjectParameters.get_exceptions_file_path()
        if exceptions_file_path.exists():
            exceptions_file_path.unlink()
        exceptions_file_path.touch()

        if ProjectParameters.get_failing_task_path_file_path().exists():
            ProjectParameters.get_failing_task_path_file_path().unlink()

        workflow_report_task = WorkflowReportTask()
        extra_data = {
            DiagnosticKeys.script_file_key: str(ProjectParameters.get_script_path().absolute()),
            DiagnosticKeys.stderr_file_key: str(ProjectParameters.get_stderr_file_path().absolute()),
            DiagnosticKeys.stdout_file_key: str(ProjectParameters.get_stdout_file_path().absolute()),
            DiagnosticKeys.exceptions_file_key: str(exceptions_file_path.absolute())
        }
        workflow_report_task.execute_start(self._workflow_start_message, extra_data)

        state, workflow_data_tree, exception_map = self.execute(
            Dict(),
            WorkflowDataTree(),
            {},  # EXECUTION PARAMETERS
            job_data,
            None
        )

        exit_status = 0
        if exception_map:
            exit_status = 1

            with exceptions_file_path.open("w") as exception_fp:
                message = "Exceptions:"
                RootJob.logger.error(message)
                print(message, file=exception_fp)
                RootJob.print_exception_map(exception_map, [], exception_fp)

            workflow_report_task.execute_finish(False, self._workflow_failure_message)
        else:
            workflow_report_task.execute_finish(True, self._workflow_success_message)

        workflow_data_tree.set_exit_status(exit_status)

        workflow_data_tree.save_to_file(workflow_data_tree_json)

        return exit_status


RootJob.logger = WorkflowUtils.get_logger(RootJob)
