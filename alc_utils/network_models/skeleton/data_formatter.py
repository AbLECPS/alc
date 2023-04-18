# Import any necessary modules here
import numpy as np


class DataFormatter:
    """The DataFormatter class provides a generic interface for formatting as-stored data types (often ROS messages)
      into numpy arrays for use with LECs. Each LEC Model defines a corresponding DataFormatter class which must
      implement the desired data formatting for the associated LEC."""

    def __init__(self, **kwargs):
        """Any required initialization should be performed here

        Args:
            **kwargs: Named arguments provided to DataFormatter.
                At a minimum, valid keyword arguments should include all fields defined in the LEC Network metadata file.
        """
        pass

    def get_topic_names(self):
        """ This function should return the list of desired ROS topic names
        If not using ROS data formats, can alternatively return a list of any type of data identifiers
        For example, with CSV files, column heading titles are often used as data identifiers"""
        return []

    def format_input(self, topics_dict):
        """This function should format ROS messages into a numpy array with the
        correct ordering & shape for input to the LEC.

        Args:
            topics_dict (dict): dictionary which maps ROS topic names to a ROS message on that topic
        """
        return None

    def format_training_output(self, topics_dict):
        """For supervised learning, This function should format ROS messages into a numpy array with the
        correct ordering & shape for labeled outputs used to train the LEC.

        Args:
            topics_dict (dict): dictionary which maps ROS topic names to a ROS message on that topic
        """
        return None
