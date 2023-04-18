from BaseCompoundJob import BaseCompoundJob
import WorkflowUtils


class SequenceJob(BaseCompoundJob):

    logger = None

    def __init__(self, parent_job, workflow_name, previous_job_name_list, next_job_name_list):
        BaseCompoundJob.__init__(self, parent_job, workflow_name, previous_job_name_list, next_job_name_list)

    def execute(self, state, workflow_data_tree, parameters, directory, parent_iteration_row):

        directory.mkdir(parents=True, exist_ok=True)

        return self.execute_iteration(state, workflow_data_tree, parameters, directory, self.get_runtime_id(), 0)


SequenceJob.logger = WorkflowUtils.get_logger(SequenceJob)
