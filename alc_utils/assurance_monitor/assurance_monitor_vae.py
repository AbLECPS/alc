# !/usr/bin/env python
# Authors:  Feiyang Cai <feiyang.cai@vanderbilt.edu>
#           Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
#           Nag Mahadevan <nag.mahadevan@vanderbilt.edu>

"""
This file implements the VAE based out of distribution detector.

- To train an out of distribution detector and compute the calibration scores,
  it uses the code in ./deep_vae.py

- To predict if the current input/ value is out of distribution,
  it uses the code in ./icad_vae.py
  Further, it computes the martingale using the three available methods - RPM, SMM, PIM.
  Current implementation of the stateful detector uses the log of SMM.

  The output of the predict method includes the 
   - N p-values, 
   - the log of maringales - RPM and SMM
   - the detector output
"""



from assurance_monitor import AssuranceMonitor
import numpy as np
from deep_vae import deepVAE
from icad_vae import ICAD
from martingales import RPM, SMM, PIM
from .detector import StatefulDetector
from scipy import stats
import os


ASSURANCE_MONITOR_NAME = "AssuranceMonitorVAE"


class AssuranceMonitorVAE(AssuranceMonitor):
    _variant = "vae"

    def __init__(self):
        super(AssuranceMonitorVAE, self).__init__()
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

        _save_dir = super(AssuranceMonitorVAE, self).save(_save_dir,
                                                               data_formatter_path,
                                                               lec_storage_metadata,
                                                               make_unique_subdir=False)

        # Return path to saved file
        return _save_dir

    def save_extra(self, save_dir):
        #save in vae_model
        self.vae_model.save_model(save_dir)
        self.vae_model = None
        self.rpm = None
        self.pim = None
        self.smm = None
        self.detector = None

    def _load_extra(self, folder_name):
        # load the vae model
        self.start = True
        self.icad_model = ICAD(folder_name)
        self.reset()
        if (not hasattr(self, 'sigma')):
            self.sigma = 15
            self.tau = 40

        self.detector = StatefulDetector(self.sigma, self.tau)

    def train(self, dataloader_training, dataloader_validation,  dataloader_calibration, model_path, **kwargs):

        # Set the mode to training
        self.vae_model = deepVAE(**kwargs)
        self.vae_model.fit(dataloader_training,dataloader_validation,
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
        vals.append(np.log(rpm))
        vals.append(np.log(smm))
        vals.append(s)
        return [vals]
