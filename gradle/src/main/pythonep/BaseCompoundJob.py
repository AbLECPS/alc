import uuid
import copy
import logging
import traceback
from pathlib import Path
from addict import Dict
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from BaseJob import BaseJob
from LaunchActivityJob import LaunchActivityJob
from LaunchExperimentJob import LaunchExperimentJob
from BranchPoint import BranchPoint
from Transform import Transform
from WorkflowDataTree import WorkflowDataTree
from WorkflowData import WorkflowData
import WorkflowUtils


class TrivialFuture:
    initial = "INITIAL"
    branch = "BRANCH"
    branch_not_taken = "BRANCH_NOT_TAKEN"
    failure = "FAILURE"

    def __init__(self, result_tuple, local_type, true_set=None):
        if true_set is None:
            true_set = set()
        self.result_tuple = result_tuple
        self.type = local_type
        self.true_set = true_set

    def result(self):
        return self.result_tuple

    def get_type(self):
        return self.type

    def get_true_set(self):
        return self.true_set


class BaseCompoundJob(BaseJob):

    logger = None

    def __init__(self, job_name, previous_job_name_list, next_job_name_list):

        BaseJob.__init__(self, job_name, previous_job_name_list, next_job_name_list)

        self.job_map = {}

    def add_input(self, input_name, job_path_list):
        raise Exception("Inapplicable method")

    def add_launch_activity_job(
            self, job_name, activity_name, activity_node, previous_job_name_list, next_job_name_list
    ):
        new_job = LaunchActivityJob(job_name, activity_name, activity_node, previous_job_name_list, next_job_name_list)
        self.job_map[job_name] = new_job

        return new_job

    def add_launch_experiment_job(self, job_name, previous_job_name_list, next_job_name_list):

        new_job = LaunchExperimentJob(job_name, previous_job_name_list, next_job_name_list)
        self.job_map[job_name] = new_job

        return new_job

    def add_branch_point(
            self, branch_point_name, previous_job_name_list, false_next_job_name_list, function, true_next_job_name_list
    ):
        new_job = BranchPoint(function, previous_job_name_list, false_next_job_name_list, true_next_job_name_list)
        self.job_map[branch_point_name] = new_job
        return new_job

    def add_transform(self, transform_name, previous_job_name_list, next_job_name_list, function):
        new_job = Transform(function, transform_name, previous_job_name_list, next_job_name_list)
        self.job_map[transform_name] = new_job
        return new_job

    def add_do_while_loop(self, loop_name, previous_job_name_list, next_job_name_list, condition_function):
        return _add_do_while_loop(self, loop_name, previous_job_name_list, next_job_name_list, condition_function)

    def add_while_loop(
            self,
            loop_name,
            previous_job_name_list,
            next_job_name_list,
            condition_function,
            parameter_update_function=None
    ):
        return _add_while_loop(
            self, loop_name, previous_job_name_list, next_job_name_list, condition_function, parameter_update_function
        )

    def add_threaded_for_each_loop(
            self, loop_name, previous_job_name_list, next_job_name_list, num_threads, parameter_updates
    ):
        return _add_threaded_for_each_loop(
            self, loop_name, previous_job_name_list, next_job_name_list, num_threads, parameter_updates
        )

    def _perform_sanity_check(self):

        previous_job_name_set_map = {}
        next_job_name_set_map = {}

        for job_name, job in self.job_map.items():
            for previous_job_name in job.get_previous_job_name_set():
                if previous_job_name not in next_job_name_set_map:
                    next_job_name_set_map[previous_job_name] = set()
                next_job_name_set_map[previous_job_name].add(job_name)

            for next_job_name in job.get_next_job_name_set():
                if next_job_name not in previous_job_name_set_map:
                    previous_job_name_set_map[next_job_name] = set()
                previous_job_name_set_map[next_job_name].add(job_name)

        start_job_name_set = set()

        # SANITY CHECK
        for job_name, job in self.job_map.items():
            previous_job_name_set = job.get_previous_job_name_set()
            if previous_job_name_set != previous_job_name_set_map.get(job_name, set()):
                message = "ERROR:  Inconsistency in previous jobs of job \"{0}\":  Specified = {1}, Calculated = {2}" \
                    .format(job_name, job.get_previous_job_name_set(), previous_job_name_set_map.get(job_name, set()))
                BaseCompoundJob.logger.error(message)
                raise Exception(message)
            if not bool(previous_job_name_set):
                start_job_name_set.add(job_name)

        end_job_name_set = set()
        for job_name, job in self.job_map.items():
            next_job_name_set = job.get_next_job_name_set()
            if next_job_name_set != next_job_name_set_map.get(job_name, set()):
                message = "ERROR:  Inconsistency in next jobs of job \"{0}\":  Specified = {1}, Calculated = {2}" \
                    .format(job_name, job.get_next_job_name_set(), next_job_name_set_map.get(job_name, set()))
                BaseCompoundJob.logger.error(message)
                raise Exception(message)
            if not bool(next_job_name_set):
                end_job_name_set.add(job_name)

        return previous_job_name_set_map, next_job_name_set_map, start_job_name_set, end_job_name_set

    @staticmethod
    def run_thread(
            job, queue, state, workflow_data_tree, execution_parameters, directory, iteration_row
    ):

        try:
            new_state, new_workflow_data_tree, exception_map = job.execute(
                state, workflow_data_tree, execution_parameters, directory, iteration_row
            )
            queue.put(job.get_job_name())

            return new_state, new_workflow_data_tree, exception_map
        except Exception as e:
            queue.put(job.get_job_name())
            return state, workflow_data_tree, {job.get_job_name(): (e, traceback.format_exc())}

    @staticmethod
    def get_ready_job_name_set(completed_job_name, next_job_name_set, dependency_job_name_set_map):

        ready_job_name_set = set()

        for next_job_name in next_job_name_set:
            dependency_job_name_set = dependency_job_name_set_map.get(next_job_name)
            dependency_job_name_set.remove(completed_job_name)

            if not bool(dependency_job_name_set):
                ready_job_name_set.add(next_job_name)

        return ready_job_name_set

    @staticmethod
    def update_result_tuple(result_tuple_to_update, new_result_tuple):
        result_tuple_to_update[0].update(new_result_tuple[0])
        result_tuple_to_update[1].merge_data(new_result_tuple[1])
        result_tuple_to_update[2].update(new_result_tuple[2])

    def finish_result_tuple(self, result_tuple):
        return (*result_tuple[:-1], {self.job_name: result_tuple[-1]}) if result_tuple[-1] else result_tuple

    @staticmethod
    def get_upstream_trivial_type(prefix, dependency_trivial_future, job_name):
        upstream_trivial_type = dependency_trivial_future.get_type()

        if upstream_trivial_type == TrivialFuture.failure:
            BaseCompoundJob.logger.info(
                "{0}: Job \"{1}\" incurred a failure or is downstream from a failed job ...".format(prefix, job_name)
            )
        elif upstream_trivial_type == TrivialFuture.branch_not_taken:
            BaseCompoundJob.logger.info(
                "{0}: Job \"{1}\" is downstream from a branch not taken ...".format(prefix, job_name)
            )
        elif upstream_trivial_type == TrivialFuture.branch and job_name not in dependency_trivial_future.get_true_set():
            BaseCompoundJob.logger.info(
                "{0}: Job \"{1}\" is part of a branch not taken ...".format(prefix, job_name)
            )
            upstream_trivial_type = TrivialFuture.branch_not_taken

        return upstream_trivial_type

    @staticmethod
    def get_combined_trivial_type(combined_trivial_type, true_trivial_type):
        if combined_trivial_type == TrivialFuture.failure or true_trivial_type == TrivialFuture.failure:
            return TrivialFuture.failure

        if combined_trivial_type == TrivialFuture.branch_not_taken or \
                true_trivial_type == TrivialFuture.branch_not_taken:
            return TrivialFuture.branch_not_taken

        return TrivialFuture.initial

    def start_jobs(
            self,
            job_name_set,
            previous_job_name_set_map,
            dependency_job_name_set_map,
            future_map,
            running_job_name_set,
            executor,
            queue,
            execution_parameters,
            iteration_directory,
            iteration_row
    ):

        iteration_num = iteration_row.get(WorkflowDataTree.index_key)
        prefix = "{0}:{1}:iteration-{2}".format(self.get_job_name(), self.get_runtime_id(), iteration_num)

        BaseCompoundJob.logger.info("{0}: starting jobs {1} ...".format(prefix, sorted(list(job_name_set))))

        while bool(job_name_set):

            job_name = list(job_name_set)[0]
            job_name_set.remove(job_name)
            job = self.job_map.get(job_name)

            BaseCompoundJob.logger.info("{0}: processing job \"{1}\" ...".format(prefix, job_name))

            dependency_job_set = previous_job_name_set_map.get(job_name, set())

            BaseCompoundJob.logger.info(
                "{0}: combining runtime data from jobs {1} in order to get the input for job \"{2}\".".format(
                    prefix, sorted(list(dependency_job_set)), job_name
                )
            )

            new_result_tuple = (Dict(), WorkflowDataTree(), {})

            combined_trivial_type = TrivialFuture.initial
            for dependency_job_name in dependency_job_set:

                BaseCompoundJob.logger.info(
                    "{0}: integrating runtime data from job \"{1}\" in order to process job \"{2}\"."
                    .format(prefix, dependency_job_name, job_name)
                )

                dependency_trivial_future = future_map.get(dependency_job_name)
                upstream_trivial_type = BaseCompoundJob.get_upstream_trivial_type(
                    prefix, dependency_trivial_future, job_name
                )
                combined_trivial_type = BaseCompoundJob.get_combined_trivial_type(
                    combined_trivial_type, upstream_trivial_type
                )

                next_result_tuple = dependency_trivial_future.result()
                BaseCompoundJob.update_result_tuple(new_result_tuple, next_result_tuple)

            if combined_trivial_type != TrivialFuture.initial:

                BaseCompoundJob.logger.info(
                    "{0}: Job \"{1}\" will not be executed, but will pass its inputs to its downstream jobs "
                    "(which will do the same)".format(prefix, job_name)
                )

                future_map[job_name] = TrivialFuture(new_result_tuple, combined_trivial_type)

                ready_branch_job_name_set = BaseCompoundJob.get_ready_job_name_set(
                    job_name, job.get_next_job_name_set(), dependency_job_name_set_map
                )
                job_name_set = job_name_set.union(ready_branch_job_name_set)

            elif isinstance(job, BranchPoint):
                BaseCompoundJob.logger.info("{0}: job \"{1}\" is a branch job.".format(prefix, job_name))

                next_job_name_set = job.get_next_job_name_set()
                ready_branch_job_name_set = BaseCompoundJob.get_ready_job_name_set(
                    job_name, next_job_name_set, dependency_job_name_set_map
                )

                try:
                    branch_job_name_set = job.execute(new_result_tuple[1])
                    not_taken_job_name_set = next_job_name_set.difference(branch_job_name_set)

                    future_map[job_name] = TrivialFuture(new_result_tuple, TrivialFuture.branch, branch_job_name_set)

                    BaseCompoundJob.logger.info("{0}: branch job \"{1}\": jobs {2} will be executed.".format(
                        prefix, job_name, sorted(list(branch_job_name_set))
                    ))
                    BaseCompoundJob.logger.info(
                        "{0}: branch job \"{1}\": jobs {2} will not be executed, but will pass-through data "
                        "to get final workflow_data_tree".format(prefix, job_name, sorted(list(not_taken_job_name_set)))
                    )

                except Exception as e:
                    new_result_tuple[-1][job_name] = (e, traceback.format_exc())

                    future_map[job_name] = TrivialFuture(new_result_tuple, TrivialFuture.failure)

                    BaseCompoundJob.logger.info(
                        "{0}: branch job \"{1}\": has failed due to an erroneous condition.".format(prefix, job_name)
                    )
                    BaseCompoundJob.logger.info(
                        "{0}: branch job \"{1}\": downstream jobs {2} will not be executed "
                        "but will pass-through data to get final workflow_data_tree".format(
                            prefix, job_name, sorted(list(next_job_name_set))
                        )
                    )

                BaseCompoundJob.logger.info("{0}: branch job \"{1}\" adding jobs {2} to start set".format(
                    prefix, job_name, sorted(list(ready_branch_job_name_set))
                ))
                job_name_set = job_name_set.union(ready_branch_job_name_set)

            else:

                future_map[job_name] = executor.submit(
                    BaseCompoundJob.run_thread,
                    job,
                    queue,
                    *new_result_tuple[:-1],
                    execution_parameters,
                    iteration_directory,
                    iteration_row
                )

                BaseCompoundJob.logger.info("{0}: adding job \"{1}\" to running jobs".format(prefix, job_name))

                running_job_name_set.add(job_name)

    def execute_iteration(
            self,
            state,
            workflow_data_tree,
            execution_parameters,
            directory,
            parent_runtime_id,
            iteration_num
    ):

        previous_job_name_set_map, next_job_name_set_map, start_job_name_set, end_job_name_set = \
            self._perform_sanity_check()

        dependency_job_name_set_map = copy.deepcopy(previous_job_name_set_map)
        future_map = {}
        running_job_name_set = set()

        if len(start_job_name_set) == 0:
            message = "ERROR:  no start jobs detected in body of compound job \"{0}\"".format(self.job_name)
            BaseCompoundJob.logger.error(message)
            raise Exception(message)

        if len(end_job_name_set) == 0:
            message = "ERROR:  no end jobs detected in body of compound job \"{0}\"".format(self.job_name)
            BaseCompoundJob.logger.error(message)
            raise Exception(message)

        iteration_name = "iteration-{0}".format(iteration_num)
        iteration_directory = Path(directory, iteration_name)
        iteration_directory.mkdir(parents=True, exist_ok=True)

        iteration_id = self.get_dynamic_id(iteration_directory)

        iteration_row = workflow_data_tree.add_iteration_data(
            iteration_name, iteration_id, iteration_num, parent_runtime_id
        )

#        workflow_data_tree.save_to_file(Path(iteration_directory, "workflow_data_tree.json"))

        workflow_data = WorkflowData(state, workflow_data_tree, iteration_row)
        updated_execution_parameters = self.get_updated_execution_parameters(workflow_data, execution_parameters)

        trivial_job_name = "trivial_{0}".format(uuid.uuid4().hex)

        future_map[trivial_job_name] = TrivialFuture((state, workflow_data_tree, {}), TrivialFuture.initial)
        for start_job_name in start_job_name_set:
            previous_job_name_set_map[start_job_name] = {trivial_job_name}

        prefix = "{0}:{1}:iteration-{2}".format(self.get_job_name(), self.get_runtime_id(), iteration_num)

        queue = Queue()

        with ThreadPoolExecutor() as executor:

            self.start_jobs(
                start_job_name_set,
                previous_job_name_set_map,
                dependency_job_name_set_map,
                future_map,
                running_job_name_set,
                executor,
                queue,
                updated_execution_parameters,
                iteration_directory,
                iteration_row
            )

            BaseCompoundJob.logger.info("{0}: waiting for jobs {1}".format(
                prefix, sorted(list(running_job_name_set))
            ))

            while bool(running_job_name_set):

                completed_job_name = queue.get()

                result_tuple = future_map.get(completed_job_name).result()
                trivial_type = TrivialFuture.initial
                if result_tuple[-1]:
                    BaseCompoundJob.logger.info(
                        "{0}: detected job \"{1}\" has completed with one or more exceptions".format(
                            prefix, completed_job_name
                        )
                    )
                    trivial_type = TrivialFuture.failure
                else:
                    BaseCompoundJob.logger.info("{0}: detected job \"{1}\" has completed".format(
                        prefix, completed_job_name
                    ))

                future_map[completed_job_name] = TrivialFuture(result_tuple, trivial_type)

                running_job_name_set.remove(completed_job_name)

                start_job_name_set = BaseCompoundJob.get_ready_job_name_set(
                    completed_job_name,
                    next_job_name_set_map.get(completed_job_name, set()),
                    dependency_job_name_set_map
                )

                self.start_jobs(
                    start_job_name_set,
                    previous_job_name_set_map,
                    dependency_job_name_set_map,
                    future_map,
                    running_job_name_set,
                    executor,
                    queue,
                    updated_execution_parameters,
                    iteration_directory,
                    iteration_row
                )

                BaseCompoundJob.logger.info("{0}: waiting for jobs {1}".format(
                    prefix, sorted(list(running_job_name_set))
                ))

        BaseCompoundJob.logger.info("{0}: completed iteration".format(prefix))

        BaseCompoundJob.logger.info(
            "{0}: combining state and workflow_data_tree from jobs {1} to get iteration output".format(
                prefix, sorted(list(end_job_name_set))
            )
        )

        new_result_tuple = (Dict(), WorkflowDataTree(), {})
        for job_name in end_job_name_set:
            future = future_map.get(job_name, None)
            if future is not None:
                BaseCompoundJob.logger.info(
                    "{0}: integrating state and workflow_data_tree from job \"{1}\" to get iteration output".format(
                        prefix, job_name
                    )
                )
                next_result_tuple = future.result()
                BaseCompoundJob.update_result_tuple(new_result_tuple, next_result_tuple)

        if not new_result_tuple[1].get_data():
            BaseCompoundJob.logger.error("{0}: ERROR -- was not able to get any data from iteration!".format(prefix))
            new_result_tuple = (
                Dict(state), WorkflowDataTree().merge_data(workflow_data_tree), new_result_tuple[2]
            )

        return new_result_tuple


BaseCompoundJob.logger = WorkflowUtils.get_logger(BaseCompoundJob)
BaseCompoundJob.logger.setLevel(logging.INFO)


from SequentialLoopJob import SequentialLoopJob


def _add_do_while_loop(self, loop_name, previous_job_name_list, next_job_name_list, condition_function):

    new_loop = SequentialLoopJob(
        self, loop_name, previous_job_name_list, next_job_name_list, condition_function, True
    )
    self.job_map[loop_name] = new_loop

    return new_loop


def _add_while_loop(
        self, loop_name, previous_job_name_list, next_job_name_list, condition_function, parameter_update_function=None
):

    new_loop = SequentialLoopJob(
        loop_name,
        previous_job_name_list,
        next_job_name_list,
        condition_function,
        False,
        parameter_update_function
    )
    self.job_map[loop_name] = new_loop

    return new_loop


from ThreadedForEachLoopJob import ThreadedForEachLoopJob


def _add_threaded_for_each_loop(
        self, loop_name, previous_job_name_list, next_job_name_list, num_threads, parameter_updates
):
    new_loop = ThreadedForEachLoopJob(
        loop_name, previous_job_name_list, next_job_name_list, num_threads, parameter_updates
    )
    self.job_map[loop_name] = new_loop

    return new_loop
