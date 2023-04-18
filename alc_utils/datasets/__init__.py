import os
import alc_utils.common as alc_common

DATASET_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))


dataset_name_to_module_name_map = {
    "rosbag": "dataset_rosbag.py",
    "ros": "dataset_rosbag.py",
    "bag": "dataset_rosbag.py",
    "rosbag_hdf5": "dataset_rosbag_hdf5.py",
    "csv": "dataset_csv.py",
    "png": "dataset_png.py"
}


# Load Dataset class from provided python module path.
# Provides functions for loading data from various storage mediums
def load_dataset_class(dataset_name):
    # Try to load interpreter by name first. If that fails, assume path was provided instead
    dataset_name_input = str(dataset_name)
    dataset_name = alc_common.normalize_string(dataset_name)
    dataset_module_path = find_dataset_module_path_by_name(dataset_name)
    if dataset_module_path is None:
        dataset_module_path = dataset_name_input
    dataset_module = alc_common.load_python_module(dataset_module_path)
    class_name = dataset_module.DATASET_NAME
    dataset_class = getattr(dataset_module, class_name)
    return dataset_class


def find_dataset_module_path_by_name(dataset_name):
    dataset_module_name = dataset_name_to_module_name_map.get(
        dataset_name, None)
    if dataset_module_name is None:
        return None
    return os.path.join(DATASET_MODULE_PATH, dataset_module_name)
