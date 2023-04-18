import operator
from addict import Dict
from Result import Result
from WorkflowDataTree import WorkflowDataTree
from ParameterUpdates import ParameterUpdates, ParameterUpdatesAux
import WorkflowUtils


class WorkflowData:

    logger = None
    parameter_values_key = "parameter_values"

    def __init__(self, state, workflow_data_tree, row):
        self.state = state
        self.workflow_data_tree = workflow_data_tree
        self.row = row

    def get_state(self):
        return self.state

    def get_workflow_data_tree(self):
        return self.workflow_data_tree.copy()

    def get_row(self):
        return self.row

    def get_value(self, data_path, function_list):
        return self.get_workflow_data_tree().get_value(data_path, function_list)

    def get_results(self, path_list):

        results_list = self.workflow_data_tree.get_path_data(path_list)
        return [Result(result) for result in results_list]

    def get_results_relative(self, *args):

        if len(args) == 1:
            iteration_specifier = -1
            path_list = args[0]
        else:
            iteration_specifier, path_list = args

        if not isinstance(path_list, list):
            path_list = [path_list]

        initial_iteration_list = self.workflow_data_tree.get_child_iteration_list(self.row, iteration_specifier)

        results_list = self.workflow_data_tree.get_path_data_relative(path_list, initial_iteration_list)
        return [Result(result) for result in results_list]

    def get_num_iterations(self):
        return WorkflowDataTree.get_num_iterations(self.row)

    @staticmethod
    def get_row_parameter_updates(row, *args):
        if WorkflowDataTree.get_row_type(row) != WorkflowDataTree.loop_type:
            return None

        if WorkflowData.parameter_values_key not in row:
            row[WorkflowData.parameter_values_key] = []
        parameter_values_list = row.get(WorkflowData.parameter_values_key)

        parameter_updates = ParameterUpdates()
        num_iterations = WorkflowDataTree.get_num_iterations(row)

        if args:
            iteration_specifier = args[0]
            if not isinstance(iteration_specifier, int):
                WorkflowData.logger.warning(
                    "WorkflowData: workflow_data.get_parameter_updates(iteration_specifier): iteration_specifier "
                    "({0}) must be an integer.  Returning empty ParameterUpdates object.".format(iteration_specifier)
                )
                return parameter_updates

            comparator = operator.le if iteration_specifier >= 0 else operator.lt
            abs_is = abs(iteration_specifier)
            if comparator(num_iterations, abs_is):
                WorkflowData.logger.warning(
                    "WorkflowData: workflow_data.get_parameter_updates(iteration_specifier): iteration specifier "
                    "({0}) is out of bounds for number of parameter updates currently in this loop ({1}). "
                    "Returning empty ParameterUpdates object".format(
                        iteration_specifier, num_iterations
                    )
                )
                return parameter_updates

            old_parameter_values = Dict(row.get(WorkflowData.parameter_values_key)[iteration_specifier])
            ParameterUpdatesAux.assign_update_map(parameter_updates, old_parameter_values)

            if len(args) > 1:
                keep_key_set = args[1]
                if not isinstance(keep_key_set, set):
                    WorkflowData.logger.warning(
                        "WorkflowData: workflow_data.get_parameter_updates(_, keep_parameter_set): keep_parameter_set "
                        "is not a set -- ignoring"
                    )
                else:
                    ParameterUpdatesAux.cull_values(parameter_updates, keep_key_set)

            return parameter_updates

        if not parameter_values_list:
            parameter_values_list.append(parameter_updates.update_map__)
            return parameter_updates

        num_parameter_values = len(parameter_values_list)
        if num_parameter_values <= num_iterations:
            parameter_values = Dict(parameter_values_list[-1])
            parameter_values_list.append(parameter_values)
            ParameterUpdatesAux.assign_update_map(parameter_updates, parameter_values)
        else:
            ParameterUpdatesAux.assign_update_map(parameter_updates, parameter_values_list[-1])

        return parameter_updates

    def get_parameter_updates(self, *args):
        return WorkflowData.get_row_parameter_updates(self.row, *args)

    def get_parent_parameter_updates(self, parameter_name_list):
        parent_iteration_row = self.workflow_data_tree.get_parent_row(self.row)
        parent_loop = self.workflow_data_tree.get_parent_row(parent_iteration_row)
        return WorkflowData.get_row_parameter_updates(parent_loop, -1, set(parameter_name_list))


WorkflowData.logger = WorkflowUtils.get_logger(WorkflowData)
