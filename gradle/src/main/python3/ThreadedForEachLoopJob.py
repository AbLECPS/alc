import logging
import concurrent.futures
from pathlib import Path
from addict import Dict
from BaseCompoundJob import BaseCompoundJob
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
import WorkflowUtils


class ThreadedForEachLoopJob(BaseCompoundJob):

    logger = None

    def __init__(self, parent_loop, loop_name, previous_job_name_list, next_job_name_list, num_threads):
        BaseCompoundJob.__init__(self, parent_loop, loop_name, previous_job_name_list, next_job_name_list)
        self.num_threads = num_threads

    def execute(self, state, workflow_data_tree, parameters, directory, parent_iteration_row):

        loop_directory = Path(directory, self.job_name)
        loop_directory.mkdir(parents=True, exist_ok=True)

        self.set_runtime_id(loop_directory)

        new_row = workflow_data_tree.add_loop_data(self, parent_iteration_row.get(WorkflowDataTree.id_key))

        workflow_data = WorkflowData(state, workflow_data_tree, new_row)

        num_threads = self.num_threads(workflow_data) \
            if callable(self.num_threads) else self.num_threads

        future_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index in range(num_threads):
                ThreadedForEachLoopJob.logger.info("ThreadedForEachLoopJob:  spawning iteration {0}".format(index))
                future_list.append(
                    executor.submit(
                        self.execute_iteration,
                        Dict(state),
                        workflow_data_tree.copy(),
                        parameters, loop_directory,
                        self.get_runtime_id(),
                        index
                    )
                )

        new_result_tuple = (Dict(), WorkflowDataTree(), {})
        for future in future_list:
            next_result_tuple = future.result()
            BaseCompoundJob.update_result_tuple(new_result_tuple, next_result_tuple)

        return self.finish_result_tuple(new_result_tuple)

ThreadedForEachLoopJob.logger = WorkflowUtils.get_logger(ThreadedForEachLoopJob)
ThreadedForEachLoopJob.logger.setLevel(logging.DEBUG)
