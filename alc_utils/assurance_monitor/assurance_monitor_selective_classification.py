# !/usr/bin/env python
# Authors:
"""This file defines the implemenation of AssuranceMonitor based on Selective Classification using Siamese network."""
from __future__ import print_function



from assurance_monitor import AssuranceMonitor
import numpy as np
from deep_selective_classification import selectiveClassification
from icad_selective_classification import ICAD
from martingales import RPM, SMM, PIM
from detector import StatefulDetector
from scipy import stats
import os

ASSURANCE_MONITOR_NAME = "AssuranceMonitorSelectiveClassification"


class AssuranceMonitorSelectiveClassification(AssuranceMonitor):
    _variant = "selective_classification"

    def __init__(self):
        super(AssuranceMonitorSelectiveClassification, self).__init__()
        self.training_model = None
        self.predictive_model = None
        self.start = True
        self.num_classes = None
        self.window_size = None
        self.a_s = None
        self.b_s = None
        self.threshold_s = None
        self.reset()

    def reset(self):
        if (self.predictive_model):
            self.predictive_model.clear_windows()

    # Override default save method
    def save(self, save_dir, data_formatter_path=None, lec_storage_metadata=None, make_unique_subdir=False):
        if make_unique_subdir:
            _save_dir = self._make_unique_subdir(save_dir)
        else:
            _save_dir = save_dir

        self.save_extra(_save_dir)

        self.training_model = None

        _save_dir = super(AssuranceMonitorSelectiveClassification, self).save(_save_dir,
                                                                              data_formatter_path,
                                                                              lec_storage_metadata,
                                                                              make_unique_subdir=False)

        # Return path to saved file
        return _save_dir

    def save_extra(self, save_dir):
        self.training_model.save_model(save_dir)

    def _load_extra(self, folder_name):
        # load the svdd model
        self.start = True
        self.predictive_model = ICAD(
            folder_name, self.num_classes)
        self.reset()

    def train(self, dataloader_training, dataloader_calibration, dataloader_testing, model_path, **kwargs):

        # Set the mode to training
        self.training_model = selectiveClassification(
            **kwargs)
        self.num_classes, self.a_s, self.b_s, self.threshold_s = self.training_model.fit(
            dataloader_training, dataloader_calibration, dataloader_testing, model_path, **kwargs)

    def _evaluate(self, input_data, predicted_output, **kwargs):
        if input_data is None:
            return []
        if (self.start):
            self.start = False
            self.predictive_model.update_snapshot_params(**kwargs)
            self.predictive_model.update_sequence_params(**kwargs)
            print('thresholds set -> snapshot: {}, combination: {}'.format(
                self.predictive_model.threshold_s, self.predictive_model.threshold_c))

        pred_decision_vals = self.predictive_model.evaluate(input_data)
        return pred_decision_vals
