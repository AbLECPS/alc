from pathlib import Path
from addict import Dict
from SequenceJob import SequenceJob
from WorkflowDataTree import WorkflowDataTree
from WorkflowReportTask import WorkflowReportTask
from ProjectParameters import ProjectParameters
import WorkflowParameters
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
        SequenceJob.__init__(self, None, "workflow", [], [])
        self.workflow_dir = workflow_dir
        self.set_is_root()

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

        job_data = Path(self.workflow_dir, "job_data")
        job_data.mkdir(parents=True, exist_ok=True)

        workflow_data_tree_dir = Path(self.workflow_dir, "data")
        workflow_data_tree_dir.mkdir(parents=True, exist_ok=True)

        workflow_data_tree_json = Path(workflow_data_tree_dir, "workflow_data_tree.json")

        exceptions_file_path = ProjectParameters.get_exceptions_file_path()
        if exceptions_file_path.exists():
            exceptions_file_path.unlink()
        exceptions_file_path.touch()

        if ProjectParameters.get_failing_task_path_file_path().exists():
            ProjectParameters.get_failing_task_path_file_path().unlink()

        report_task_dir_path = Path(self.workflow_dir, RootJob._report_task_dir_name)
        workflow_report_task = WorkflowReportTask(
            report_task_dir_path,
            RootJob._workflow_path,
            RootJob._workflow_start_message,
            RootJob._workflow_success_message
        )
        workflow_report_task.set_extra_data({
            DiagnosticKeys.script_file_key: str(ProjectParameters.get_script_path().absolute()),
            DiagnosticKeys.stderr_file_key: str(ProjectParameters.get_stderr_file_path().absolute()),
            DiagnosticKeys.stdout_file_key: str(ProjectParameters.get_stdout_file_path().absolute()),
            DiagnosticKeys.exceptions_file_key: str(exceptions_file_path.absolute())
        })
        workflow_report_task.execute()

        state, workflow_data_tree, exception_map = self.execute(
            Dict(),
            WorkflowDataTree(),
            WorkflowParameters.workflow_parameters,
            job_data,
            None
        )

        workflow_data_tree.save_to_file(workflow_data_tree_json)

        exit_status = 0
        if exception_map:
            workflow_report_task.set_failure(RootJob._workflow_failure_message)
            exit_status = 1

            with exceptions_file_path.open("w") as exception_fp:
                message = "Exceptions:"
#                RootJob.logger.error(message)
                print(message, file=exception_fp)
                RootJob.print_exception_map(exception_map, [], exception_fp)

            if ProjectParameters.get_failing_task_path_file_path().exists():
                with ProjectParameters.get_failing_task_path_file_path().open("r") as failing_task_path_file_fp:
                    failing_task_path = failing_task_path_file_fp.read().strip()
                workflow_report_task.set_extra_data({DiagnosticKeys.failing_task_path_key: failing_task_path})

        workflow_report_task.execute()

        return exit_status


RootJob.logger = WorkflowUtils.get_logger(RootJob)
