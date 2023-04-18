#!/usr/bin/env python
""" This file provides a skeleton for defining custom PyTorch Dataset classes.

Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu> """

# Add any additional import statements as needed
from torch.utils import data
import alc_utils.config as alc_config

# Provided DATASET_NAME must match class name exactly. Not recommended to use example "MyDataset" name
DATASET_NAME = "MyDataset"


# Recommend using the torch.utils.data.Dataset class whenever possible for feature-complete compatibility with ALC.
# Depending on your dataset, torch.utils.data.IterableDataset can also be used. However, not all ALC features will work.
# See PyTorch data loading documentation for more info: https://pytorch.org/docs/stable/data.html
class MyDataset(data.Dataset):
    """Custom PyTorch Dataset class skeleton for ALC Toolchain.
    Recommended to place a short description of the purpose of your custom Dataset here.

    See PyTorch data loading documentation for additional details as needed: https://pytorch.org/docs/stable/data.html
    """

    def __init__(self, data_dir_list, data_formatter, **kwargs):
        super(MyDataset, self).__init__()

        # Save arguments
        self.formatter = data_formatter
        self.topic_names = self.formatter.get_topic_names()
        self.data_dir_list = data_dir_list
        self.kwargs = kwargs

        # Read any optional key-word arguments and perform any additional setup here.
        # Example: self.rng_seed = kwargs.get("rng_seed", alc_config.training_defaults.RNG_SEED)

        # Generally, Dataset class should search the provided directories (self.data_dir_list) for any relevant data
        # files. For small datasets, these files can be loaded directly into memory and indexed using typical array
        # indexing. For larger datasets which will not fit in memory, recommended to build an index of the available
        # data and dynamically load individual data-points when __getitem__ is called. Certain existing data formats
        # provide better dynamic loading capabilities (eg. HDF5) than others (eg. ROS Bag).

    def __len__(self):
        """Return the length of the loaded dataset. Function not required if using IterableDataset class."""
        pass

    def __getitem__(self, idx):
        """Fetch and return the training data point corresponding to index 'idx'.
        Data point should be returned as a 2-tuple in the form (input_data_point, output_data_point).
        Function not required if using IterableDataset class."""

        # Data must be properly formatted for training before returning from __getitem__.
        # See DataFormatter skeleton for usage details. General usage example:
        #
        # formatted_input_data = self.formatter.format_input(<raw_data_point>)
        # formatted_output_data = self.formatter.format_training_output(<raw_data_point>)
        # return formatted_input_data, formatted_output_data
        pass

    def __del__(self):
        """Perform any necessary cleanup when this class is deleted."""
        pass

    # Uncomment this function if using torch.utils.data.IterableDataset type dataset. Otherwise, can be removed.
    # def __iter__(self):
    #     """Return the next data point in the loaded dataset.
    #     Data point should be returned as a 2-tuple in the form (input_data_point, output_data_point).
    #     Function only required if using IterableDataset class"""
    #     pass

    # Define additional functions here as needed.
