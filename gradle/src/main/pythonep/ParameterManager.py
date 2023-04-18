from addict import Dict


class ParameterManager:

    _all_specifier = "all"

    def __init__(self):
        self.parameter_map = {}
        self.num_combinations = 1

    def add_parameter(self, name, value_list):
        value_sure_list = list(value_list)
        self.num_combinations *= len(value_sure_list)
        self.parameter_map[name] = value_sure_list

    def is_complete(self, workflow_data):
        return workflow_data.get_num_iterations() >= self.num_combinations if self.parameter_map else True

    def get_num_combinations(self):
        return self.num_combinations if self.parameter_map else 0

    def get_combination(self, workflow_data):

        workflow_data_tree = workflow_data.get_workflow_data_tree()
        row = workflow_data.get_row()

        iteration_num = workflow_data_tree.get_iteration_num(row)

        parameter_updates = Dict()
        divisor = 1
        for name, value_list in self.parameter_map.items():
            modulus = len(value_list)
            index = (iteration_num // divisor) % modulus
            divisor *= modulus

            parameter_updates[name] = value_list[index]

        return parameter_updates
