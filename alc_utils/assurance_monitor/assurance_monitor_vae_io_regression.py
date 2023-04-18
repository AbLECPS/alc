# !/usr/bin/env python
# Authors:  Feiyang Cai <feiyang.cai@vanderbilt.edu>
#           Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
#           Nag Mahadevan <nag.mahadevan@vanderbilt.edu>
"""This file defines the implemenation of AssuranceMonitor based on VAE using both input and output for regression."""
from __future__ import print_function

from assurance_monitor import AssuranceMonitor
import numpy as np
from deep_vae_io_regression import deepVAE
from icad_vae_io_regression import ICAD
from assurance_monitor_svdd_vae.martingales import RPM, SMM, PIM
from scipy import stats
import os
from detector import StatefulDetector


ASSURANCE_MONITOR_NAME = "AssuranceMonitorVAEIORegression"


class AssuranceMonitorVAEIORegression(AssuranceMonitor):
    _variant = "vae_io_regression"

    def __init__(self):
        super(AssuranceMonitorVAEIORegression, self).__init__()
        self.epsilon = 0.75
        self.window_size = 10
        self.rpm = None
        self.pim = None
        self.smm = None
        self.vae_model = None
        self.icad_model = None
        self.start = True
        self.sigma = 15
        self.tau = 40
        self.reset()
        self.detector = StatefulDetector(self.sigma, self.tau)

    def reset(self):
        self.epsilon = 0.75
        self.window_size = 10
        self.rpm = RPM(self.epsilon, self.window_size)
        self.pim = PIM(self.window_size)
        self.smm = SMM(self.window_size)

    # Override default save method
    def save(self, save_dir, data_formatter_path=None, lec_storage_metadata=None, make_unique_subdir=False):
        if make_unique_subdir:
            _save_dir = self._make_unique_subdir(save_dir)
        else:
            _save_dir = save_dir

        self.save_extra(_save_dir)

        _save_dir = super(AssuranceMonitorVAEIORegression, self).save(_save_dir,
                                                                      data_formatter_path,
                                                                      lec_storage_metadata,
                                                                      make_unique_subdir=False)

        # Return path to saved file
        return _save_dir

    def save_extra(self, save_dir):
        #save in svdd_model
        self.vae_model.save_model(save_dir)
        self.vae_model = None
        self.rpm = None
        self.pim = None
        self.smm = None
        self.detector = None

    def _load_extra(self, folder_name):
        # load the svdd model
        self.start = True
        self.icad_model = ICAD(folder_name)
        self.reset()
        if (not hasattr(self, 'sigma')):
            self.sigma = 15
            self.tau = 40
        self.detector = StatefulDetector(self.sigma, self.tau)

    def train(self, dataloader_training, dataloader_calibration, model_path, **kwargs):

        # Set the mode to training
        self.vae_model = deepVAE(**kwargs)
        self.vae_model.fit(dataloader_training,
                           dataloader_calibration, model_path, **kwargs)

    def _evaluate(self, input_data, predicted_output, **kwargs):
        if (self.start):
            am_threshold = kwargs.get('am_detector_threshold', None)
            if (am_threshold):
                self.sigma = am_threshold[0]
                self.tau = am_threshold[1]
                self.detector.update_threshold(am_threshold)
            self.start = False
        if input_data is None:
            self.reset()
            print('input data is none')
            return []
        self.reset()
        vals = []
        for i in range(0, self.window_size):
            p = self.icad_model.evaluate(input_data)
            vals.append(p)
            rpm = self.rpm(p)
            smm = self.smm(p)
            pim = self.pim(p)
        s, _ = self.detector(np.log(smm))
        vals.append(rpm)
        vals.append(smm)
        vals.append(s)
        return [vals]
