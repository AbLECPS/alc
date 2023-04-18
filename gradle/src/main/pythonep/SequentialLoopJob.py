from pathlib import Path
from addict import Dict
import logging
from BaseCompoundJob import BaseCompoundJob
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
import WorkflowUtils


class SequentialLoopJob(BaseCompoundJob):

    logger = None

    def __init__(
            self,
            loop_name,
            previous_job_name_list,
            next_job_name_list,
            condition_function,
            is_do_while,
            parameter_update_function=None
    ):
        BaseCompoundJob.__init__(self, loop_name, previous_job_name_list, next_job_name_list)

        self.condition_function = condition_function
        self.parameter_update_function = parameter_update_function
        self.is_do_while = is_do_while

    def execute(self, state, workflow_data_tree, execution_parameters, directory, parent_iteration_row):

        loop_directory = Path(directory, self.job_name)
        loop_directory.mkdir(parents=True, exist_ok=True)

        self.set_runtime_id(loop_directory)

        new_row = workflow_data_tree.add_loop_data(self, parent_iteration_row.get(WorkflowDataTree.id_key))

        # workflow_data_tree.save_to_file(Path(loop_directory, "workflow_data_tree.json"))

        iteration_num = 0

        workflow_data = WorkflowData(state, workflow_data_tree, new_row)
        condition = True if self.is_do_while else self.condition_function(workflow_data)

        new_result_tuple = (Dict(state), WorkflowDataTree().merge_data(workflow_data_tree), {})
        while condition:
            if callable(self.parameter_update_function):
                self.parameter_update_function(workflow_data)

            new_result_tuple = self.execute_iteration(
                *new_result_tuple[:-1],
                execution_parameters,
                loop_directory,
                self.get_runtime_id(),
                iteration_num
            )

            if new_result_tuple[-1]:
                break

            iteration_num += 1
            workflow_data = WorkflowData(*new_result_tuple[:-1], new_row)
            condition = self.condition_function(workflow_data)

        return self.finish_result_tuple(new_result_tuple)


SequentialLoopJob.logger = WorkflowUtils.get_logger(SequentialLoopJob)
SequentialLoopJob.logger.setLevel(logging.DEBUG)
