import os
import alc_utils.common as alc_common
import alc_utils.config as alc_config
from library_adapter_base import LibraryAdapterBase

_library_adapter_name_to_file_map = {
    "keras": os.path.join(alc_config.ML_LIBRARY_ADAPTERS_DIR, "keras_library_adapter.py"),
    "pytorch_semseg": os.path.join(alc_config.ML_LIBRARY_ADAPTERS_DIR, "pytorch_semseg_adapter.py")
}


# Load Dataset class from provided python module path.
# Provides functions for loading data from various storage mediums
def load_library_adapter(adapter_name):
    # Try to load adapter by name first. If that fails, assume path was provided instead
    adapter_name_input = str(adapter_name)
    adapter_name = alc_common.normalize_string(adapter_name)
    adapter_module_path = _library_adapter_name_to_file_map.get(
        adapter_name, None)
    if adapter_module_path is None:
        adapter_module_path = adapter_name_input
    adapter_module = alc_common.load_python_module(adapter_module_path)
    class_name = adapter_module.ADAPTER_NAME
    adapter_class = getattr(adapter_module, class_name)
    return adapter_class()
