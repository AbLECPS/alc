import json
import logging
import shutil
from pathlib import Path
from KeysAndAttributes import Keys


class ActivityInterpreterBase:

    logger = logging.Logger("foo")

    input_file_name = "launch_activity_output.json"
    original_input_file_name = "launch_activity_output.orig.json"
    inputs_dir_name = "Inputs"
    input_json_file_name = "input.json"
    modify_parameters_file_name = "parameters.json"

    modification_type_key = "modification_type"
    data_key = "Data"

    replace_modification_type = "REPLACE"
    extend_modification_type = "EXTEND"

    class ReturnValue:
        def __init__(self, message="NOT IMPLEMENTED", warnings=None, errors=None, return_code=-1):
            if errors is None:
                errors = []
            if warnings is None:
                warnings = []

            self.message = message
            self.warnings = warnings
            self.errors = errors
            self.return_code = return_code

    NOT_IMPLEMENTED = ReturnValue()
    SUCCESS = ReturnValue(message="SUCCESS", return_code=0)
    FAILED = ReturnValue(message="FAILED", return_code=1)

    def modify_inputs(self):
        # MODIFY JSON INPUT FROM 'self.input_file_name' USING "Inputs" DIRECTORY
        inputs_dir_path = Path(self.input_dir_path, self.inputs_dir_name)
        input_json_map = {}
        if not inputs_dir_path.is_dir():
            return False

        for input_json_dir_path in inputs_dir_path.iterdir():
            if not input_json_dir_path.is_dir():
                continue

            input_json_path = Path(input_json_dir_path,
                                   self.input_json_file_name)
            if not input_json_path.is_file():
                continue

            with input_json_path.open("r") as input_json_fp:
                input_json = json.load(input_json_fp)

            input_dir_name = input_json_dir_path.name
            input_json_map[input_dir_name] = input_json

        inputs_map = self.input_map.get(Keys.inputs_key_name)

        for input_name, input_json in input_json_map.items():

            input_map = inputs_map.get(input_name, None)
            if input_map is None:
                continue

            input_modification_type = input_map.get(
                self.modification_type_key, self.replace_modification_type)

            if input_modification_type == self.replace_modification_type:
                input_map[self.data_key] = input_json
            else:
                input_data = input_map.get(self.data_key)
                input_data.extend(input_json)

        return True

    def modify_parameters(self):

        modify_parameters_file_path = Path(
            self.input_dir_path, self.modify_parameters_file_name)

        if modify_parameters_file_path.is_file():
            with modify_parameters_file_path.open("r") as modify_parameters_fp:
                modify_parameters_json = json.load(modify_parameters_fp)

            parameters_section = self.input_map.get(
                Keys.parameters_key_name, None)
            if parameters_section is None:
                return False

            parameter_map_section = self.input_map.get(
                Keys.parameter_map_key_name, {})

            for key, value in modify_parameters_json.items():
                if key not in parameter_map_section:
                    self.logger.warning(
                        "key \"{0}\" not found in parameter_map".format(key))
                    continue

                json_path = parameter_map_section.get(key)
                if not isinstance(json_path, list):
                    json_path = [json_path]

                if len(json_path) == 0:
                    self.logger.warning(
                        "key \"{0}\" in parameter_map has empty path".format(key))
                    continue

                local_parameters_section = parameters_section
                parent_json_path = json_path[:-1]
                found = True
                for item in parent_json_path:
                    if item not in local_parameters_section:
                        self.logger.warning(
                            "item \"{0}\" of parameter path {1} does not exist.  Skipping path".format(
                                item, json_path)
                        )
                        found = False
                        break

                    local_parameters_section = local_parameters_section.get(
                        item)

                if not found:
                    continue

                json_leaf = json_path[-1]

                if json_leaf not in local_parameters_section:
                    self.logger.warning(
                        "item \"{0}\" of parameter path {1} does not exist.  Skipping path".format(
                            json_leaf, json_path)
                    )

                local_parameters_section[json_leaf] = value

        return True

    def __init__(self, input_dir_path):
        self.input_dir_path = Path(input_dir_path)

        input_file_path = Path(self.input_dir_path, self.input_file_name)
        #print(input_file_path)
        with input_file_path.open("r") as input_fp:
            self.input_map = json.load(input_fp)
        #print(self.input_map)


        # DO NOT CHANGE CODE BELOW
        # IT IS MADE THIS WAY TO ENSURE BOTH "self.modify_inputs()" AND "self.modify_parameters()" ARE BOTH CALLED.
        # IF THEY ARE PLACED DIRECTLY IN THE DISJUNCTION CONDITION OF THE "if", IF "self.modify_inputs()" RETURNS
        # "True", "self.modify_parameters()" WON'T BE EXECUTED.
        inputs_condition = self.modify_inputs()
        parameters_condition = self.modify_parameters()
        #print('after modifying')
        #print(self.input_map)
        if inputs_condition or parameters_condition:
            original_input_file_path = Path(
                self.input_dir_path, self.original_input_file_name)
            shutil.move(str(input_file_path), str(original_input_file_path))
            with input_file_path.open("wb") as input_fp:
                output = json.dumps(self.input_map,indent=4, sort_keys=True)
                #json.dump(self.input_map, input_fp, indent=4, sort_keys=True)
                input_fp.write(output.encode())

    def constraints(self):
        return self.NOT_IMPLEMENTED

    def setup(self):
        return self.NOT_IMPLEMENTED

    def execute(self):
        return self.NOT_IMPLEMENTED

    def result(self):
        return self.NOT_IMPLEMENTED

    def abort(self):
        pass


def get_logger(clazz):
    logger_name = clazz.__module__ + "." + clazz.__name__
    logger = logging.Logger(logger_name)
    logger.addHandler(logging.StreamHandler())
    return logger


ActivityInterpreterBase.logger = get_logger(ActivityInterpreterBase)
