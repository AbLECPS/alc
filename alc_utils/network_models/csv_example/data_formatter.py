# Simple DataFormatter for testing CSV files
import numpy as np


class DataFormatter:
    def __init__(self, input_shape=None):
        self.topic_names = ["field.feedback.state_vector0",
                            "field.feedback.state_vector1",
                            "field.feedback.state_vector2",
                            "field.feedback.state_vector3",
                            "field.feedback.state_vector7"]

        if input_shape is None:
            raise ValueError(
                "DataFormatter for CSV example requires optional parameter 'input_shape'")
        self._input_shape = input_shape

        if len(self._input_shape) > 1:
            self.model_input_cnt = input_shape[1]
        else:
            self.model_input_cnt = input_shape[0]

    def get_topic_names(self):
        return self.topic_names

    def format_input(self, topics_dict):
        input_array = np.zeros(self.model_input_cnt)
        for i in range(0, self.model_input_cnt):
            topic = self.topic_names[i]
            input_array[i] = topics_dict[topic]

        return [input_array]

    def format_training_output(self, topics_dict, input_shape):
        output_topic = self.topic_names[-1]
        training_output = np.asarray([topics_dict[output_topic]])
        return [training_output]
