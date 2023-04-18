# !/usr/bin/env python
# Authors:  Feiyang Cai <feiyang.cai@vanderbilt.edu>
#           Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
#           Nag Mahadevan <nag.mahadevan@vanderbilt.edu>
"""This file defines the SVDD AssuranceMonitor implementation."""
from __future__ import print_function

from assurance_monitor import AssuranceMonitor
import numpy as np
from deep_lrp_torch import deepLRP
from icad_lrp_torch import ICAD
from assurance_monitor_svdd_vae.martingales import RPM, SMM, PIM
from scipy import stats
import os
from detector import StatefulDetector


ASSURANCE_MONITOR_NAME = "AssuranceMonitorLRPTorch"


class AssuranceMonitorLRPTorch(AssuranceMonitor):
    _variant = "vae"

    def __init__(self):
        super(AssuranceMonitorLRPTorch, self).__init__()
        self.epsilon = 0.75
        self.window_size = 10
        self.rpm = None
        self.pim = None
        self.smm = None
        self.salmap_model = None
        self.icad_model = None
        self.num_outputs = 4
        self.norm = [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
        self.detectors = []
        self.relevance = []
        self.start = True
        self.sigma = 10
        self.tau = 30
        self.reset()

    def reset(self):
        self.epsilon = 0.75  # 0.5
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

        _save_dir = super(AssuranceMonitorLRPTorch, self).save(_save_dir,
                                                               data_formatter_path,
                                                               lec_storage_metadata,
                                                               make_unique_subdir=False)

        # Return path to saved file
        return _save_dir

    def save_extra(self, save_dir):
        #save in svdd_model
        self.lrp_model.save_model(save_dir)
        self.lrp_model = None
        self.rpm = None
        self.pim = None
        self.smm = None
        self.detector = None

    def _load_extra(self, folder_name):
        # load the svdd model
        self.start = True
        self.icad_model = ICAD(folder_name, self.norm, self.num_outputs)
        self.reset()
        if (not hasattr(self, 'sigma')):
            self.sigma = 10
            self.tau = 30
        for i in range(self.num_outputs):
            self.detectors.append(StatefulDetector(self.sigma, self.tau))

    def train(self, dataloader_training, dataloader_calibration, model_path, **kwargs):

        # Set the mode to training
        self.num_outputs = kwargs.get('num_outputs', 4)
        self.norm = kwargs.get(
            'norm', [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0])
        self.lrp_model = deepLRP(**kwargs)
        network_path = kwargs.get('network_path', None)
        if (not network_path):
            WORKING_DIRECTORY = os.getenv('ALC_WORKING_DIR')
            network_path = os.path.join(
                WORKING_DIRECTORY, 'jupyter', 'lec1_weights', 'saved_weights', 'normal')
        self.lrp_model.fit(dataloader_training, dataloader_calibration,
                           self.num_outputs, self.norm, model_path, network_path, **kwargs)

    def get_relevance(self):
        return self.relevance

    def _evaluate(self, input_data, predicted_output, **kwargs):
        #print (' self.epison ', str(self.epsilon))
        if (self.start):
            am_threshold = kwargs.get('am_detector_threshold', None)
            if (am_threshold):
                self.sigma = am_threshold[0]
                self.tau = am_threshold[1]
                self.detectors = []
                for i in range(self.num_outputs):
                    self.detectors.append(
                        StatefulDetector(self.sigma, self.tau))
            self.start = False
        if input_data is None:
            self.reset()
            print('input data is none')
            return []
        self.reset()
        vals = []
        smm = []
        rpm = []
        m = []
        pvals = []
        svals = []
        smmvals = []
        rpmvals = []
        for i in range(self.num_outputs):
            smm.append(SMM(self.window_size))
            rpm.append(RPM(self.epsilon, self.window_size))
            m.append(0)
            smmvals.append(0)
            rpmvals.append(0)
            pvals.append([])

        for i in range(0, self.window_size):
            p, relevance, y = self.icad_model.evaluate(input_data)
            # compute smm for each output's p value
            for j in range(self.num_outputs):
                smmvals[j] = smm[j](p[j])
                rpmvals[j] = rpm[j](p[j])
                m[j] = np.log(smmvals[j])
                pvals[j].append(p[j])

        for i in range(self.num_outputs):
            s, _ = self.detectors[i](m[i])
            svals.append(s)
            vals.append([])
            vals[-1].extend(pvals[i])
            vals[-1].append(rpmvals[i])
            vals[-1].append(smmvals[i])
            vals[-1].append(svals[i])

        return vals
