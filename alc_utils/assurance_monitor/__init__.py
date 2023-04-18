import os
import alc_utils.common as alc_common
from .assurance_monitor import AssuranceMonitor

ASSURANCE_MONITOR_PACKAGE = "alc_utils.assurance_monitor"

am_name_to_module_name_map = {
    "svdd": "assurance_monitor_svdd",
    "vae": "assurance_monitor_vae",
    "svdd_vae": "assurance_monitor_vae_torch",
    "salmap": "assurance_monitor_salmap",
    "lrp": "assurance_monitor_lrp",
    "vae_regression": "assurance_monitor_vae_regression",
    "vae_io_regression": "assurance_monitor_vae_io_regression",
    "selective_classification": "assurance_monitor_selective_classification",
    "multi": "multi_assurance_monitor"
}


# Load AssuranceMonitor class from provided name or python module path.
def load_assurance_monitor(assurance_monitor_name):
    # Try to load interpreter by name first. If that fails, assume path was provided instead
    assurance_monitor_name_input = str(assurance_monitor_name)
    assurance_monitor_name = alc_common.normalize_string(
        assurance_monitor_name)
    am_module_path = find_am_module_path_by_name(assurance_monitor_name)
    if am_module_path is None:
        am_module_path = assurance_monitor_name_input
    am_module = alc_common.load_python_module(am_module_path)
    class_name = am_module.ASSURANCE_MONITOR_NAME
    am_class = getattr(am_module, class_name)
    return am_class()


def find_am_module_path_by_name(am_name):
    am_module_name = am_name_to_module_name_map.get(am_name, None)
    if am_module_name is None:
        return None
    return "%s.%s" % (ASSURANCE_MONITOR_PACKAGE, am_module_name)
