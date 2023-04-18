#!/usr/bin/python3
import os
import numpy as np
from scipy import stats
import torch

"""

This code contains the code for prediction of out of distribution detection based on 
  - trained svdd network and calibration score

"""


class ICAD():
    def __init__(self, folder_path):
        self.model_path = folder_path
        self.mse = torch.nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        print(('model path', self.model_path))
        svdd_model_path = os.path.join(self.model_path, "deepSVDD.pt")
        svdd_center_path = os.path.join(self.model_path, "svdd_c.npy")
        svdd_nc_path = os.path.join(self.model_path, "svdd_nc.npy")
        svdd_network_path = os.path.join(self.model_path, "am_net.py")
        print(('svdd_network_path', svdd_network_path))

        
        if (os.path.exists(svdd_network_path)):
            import imp
            foo = imp.load_source('am_net', svdd_network_path)
            if 'SVDDNet' in dir(foo):
                self.net = foo.SVDDNet()
            elif 'SVDD' in dir(foo):
                self.net = foo.SVDD()
        if (not self.net):
                raise ValueError(
                    'No SVDDNet or SVDD  class definition found in assurance monitor network definition')
        self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load(
            svdd_model_path, map_location=torch.device('cpu')))
        self.net.eval()
        self.center = np.load(svdd_center_path, allow_pickle=True)
        self.calibration_NC = np.load(svdd_nc_path, allow_pickle=True)
        
    
    def evaluate(self, input):
        input_torch = input
        if (not torch.is_tensor(input_torch)):
            input_torch = torch.from_numpy(input_torch).float()
        input_torch = input_torch.to(self.device)
        with torch.no_grad():
            output = self.net(input_torch)
        rep = output.cpu().data.numpy()
        dist = np.sum((rep - self.center)**2, axis=1)
        print(dist)
        print(self.calibration_NC)
        p = (100 - stats.percentileofscore(self.calibration_NC, dist))/float(100)
        return p
    