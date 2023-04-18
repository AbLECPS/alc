import inspect
from future.utils import viewitems


def build_var_dict(_object_globals_dict, convert_case=None):
    """Build key-value dict of non-callable, non-private, non-module static variable names and their values.
    Optionally convert variables names to UPPER/lower case."""
    if convert_case is not None:
        convert_case = convert_case.lower()
    var_dict = {}
    for attr_name, value in viewitems(_object_globals_dict):
        if (not attr_name.startswith('_')) and (not callable(value)) and (not inspect.ismodule(value)):
            if convert_case == "upper":
                var_dict[attr_name.upper()] = value
            elif convert_case == "lower":
                var_dict[attr_name.lower()] = value
            else:
                var_dict[attr_name] = value

    return var_dict
