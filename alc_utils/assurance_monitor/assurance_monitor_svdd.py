# !/usr/bin/env python
# Authors:  Feiyang Cai <feiyang.cai@vanderbilt.edu>
#           Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
#           Nag Mahadevan <nag.mahadevan@vanderbilt.edu>


"""
This file implements the SVDD based out of distribution detector.

- To train an out of distribution detector and compute the calibration scores,
  it uses the code in ./deep_svdd.py

- To predict if the current input/ value is out of distribution,
  it uses the code in ./icad_svdd.py
  Further, it computes the martingale using the three available methods - RPM, SMM, PIM.
  Current implementation of the stateless detector uses the log of SMM.

  The output of the predict method includes the 
   - current p-value, 
   - the log of maringales - RPM and SMM computed based on the N previous windows
   - the detector output - for the current log Martingale value
"""


from .assurance_monitor import AssuranceMonitor
import numpy as np
from .deep_svdd import deepSVDD
from .icad_svdd import ICAD
from .martingales import RPM, SMM, PIM
from scipy import stats
import os
from .detector import StatelessDetector


ASSURANCE_MONITOR_NAME = "AssuranceMonitorSVDD"


class AssuranceMonitorSVDD(AssuranceMonitor):
    _variant = "svdd"

    def __init__(self):
        super(AssuranceMonitorSVDD, self).__init__()
        self.epsilon = 0.75
        self.window_size = 1
        self.rpm = None
        self.pim = None
        self.smm = None
        self.svdd_model = None
        self.icad_model = None
        self.start = True
        self.sigma = 15
        self.tau = 10
        self.reset()
        self.detector = StatelessDetector(self.tau)

    def reset(self):
        self.epsilon = 0.75
        self.window_size = 1
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

        _save_dir = super(AssuranceMonitorSVDD, self).save(_save_dir,
                                                                data_formatter_path,
                                                                lec_storage_metadata,
                                                                make_unique_subdir=False)

        # Return path to saved file
        return _save_dir

    def save_extra(self, save_dir):
        #save in svdd_model
        self.svdd_model.save_model(save_dir)
        self.svdd_model = None
        self.rpm = None
        self.pim = None
        self.smm = None
        self.detector = None

    def _load_extra(self, folder_name):
        # load the svdd model
        self.start = True
        self.icad_model = ICAD(folder_name)
        self.reset()
        if (not hasattr(self, 'tau')):
            self.tau = 10
        self.detector = StatelessDetector(self.tau)

    def train(self, dataloader_training, dataloader_validation, dataloader_calibration, model_path, **kwargs):

        # Set the mode to training
        self.svdd_model = deepSVDD(**kwargs)
        self.svdd_model.fit(dataloader_training, dataloader_validation,
                            dataloader_calibration, model_path, **kwargs)

    def _evaluate(self, input_data, predicted_output, **kwargs):
        if (self.start):
            am_threshold = kwargs.get('am_detector_threshold', None)
            if (am_threshold):
                self.tau = am_threshold[len(am_threshold)-1]
                self.detector.update_threshold(am_threshold)
            self.start = False

        if input_data is None:
            self.reset()
            print('input data is none')
            return []
        vals = []
        print (self.window_size)
        
        p = self.icad_model.evaluate(input_data)
        vals.append(p)
        rpm = self.rpm(p)
        smm = self.smm(p)
        pim = self.pim(p)
        s = 1 if self.detector(np.log(smm)) else 0
        vals.append(rpm)
        vals.append(np.log(smm))
        vals.append(s)
        return [vals]
