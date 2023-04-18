from __future__ import print_function

import numpy as np

AM_TRAINING_TOPIC = "/follow_lec_0/feedback"


class DataFormatter:
    def __init__(self):
        self.topics = [AM_TRAINING_TOPIC]
        self.cpa = -60.0
        self.ttd = 0
        self.pr = 100
        self.mode = 'TRAINING'

    def format_input(self, topics_dict):
        feedback_msg = topics_dict[AM_TRAINING_TOPIC]
        inputs = []
        normalizer = [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
        self.cpa = feedback_msg.feedback.state_vector[6]
        if self.cpa >= 0.0:
            return None
        if self.mode == 'TRAINING':
            self.ttd = feedback_msg.feedback.state_vector[3]
            self.pr = max(abs(feedback_msg.feedback.state_vector[1]), abs(
                feedback_msg.feedback.state_vector[2]))
            print("IN TRAINING MODE: ttd = ", str(self.ttd), "pr = ", self.pr)
            if self.ttd > 1:
                return None
            if self.pr < 1:
                return None
            print("DATA WAS INPUT")

        for j in range(8):
            inputs.append(
                feedback_msg.feedback.state_vector[j] / normalizer[j])
        if inputs[6] < 0:
            inputs[6] = -1
        if inputs[7] < 0:
            inputs[7] = -1
        return np.array(inputs)

    def format_training_output(self, topics_dict):
        feedback_msg = topics_dict[AM_TRAINING_TOPIC]
        outputs = []
        if self.cpa >= 0.0:
            return None
        if self.mode == 'TRAINING':
            if self.ttd > 1:
                return None
            if self.pr < 1:
                return None

        outputs.append(feedback_msg.feedback.lec1_output_action[0])
        outputs.append(feedback_msg.feedback.lec1_output_action[1])
        return np.array(outputs)

    def get_topic_names(self):
        return self.topics
