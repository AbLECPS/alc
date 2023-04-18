import alc_utils.datasets


def test_load_dataset_class():
    dataset_name_to_module_name_map = {
        "rosbag": "dataset_rosbag.py",
        "ros": "dataset_rosbag.py",
        "bag": "dataset_rosbag.py",
        "rosbag_hdf5": "dataset_rosbag_hdf5.py",
        "csv": "dataset_csv.py",
        "png": "dataset_png.py"
    }

    for key in dataset_name_to_module_name_map:
        dataset_class = alc_utils.datasets.load_dataset_class(key)
        assert(dataset_name_to_module_name_map[key] != dataset_class)
