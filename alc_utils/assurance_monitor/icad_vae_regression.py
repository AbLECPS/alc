#!/usr/bin/python3
import os
import numpy as np
from scipy import stats
import torch


class ICAD():
    def __init__(self, folder_path):
        self.model_path = folder_path
        print('model path', self.model_path)
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        nc_calibration_path = os.path.join(self.model_path, "vae_nc.npy")
        self.mse = torch.nn.MSELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if (os.path.exists(vae_model_path)):
            import imp
            network_path = os.path.join(self.model_path, "am_net.py")
            if (not os.path.exists(network_path)):
                import am_net_vae
                self.vae_net = am_net_vae.VAE()
            else:
                mods = imp.load_source('am_net', network_path)
                if ('VAE' in dir(mods)):
                    self.vae_net = mods.VAE()
                elif ('VAENet' in dir(mods)):
                    self.vae_net = mods.VAENet()
                else:
                    raise ValueError(
                        'No VAENet  class definition found in assurance monitor network definition')
            self.vae_net = self.vae_net.to(self.device)
            self.vae_net.load_state_dict(torch.load(
                vae_model_path, map_location=self.device))
            self.vae_net.eval()
            self.nc_calibration = np.load(
                nc_calibration_path, allow_pickle=True)

    def evaluate(self, input):
        input_torch = input
        if (not torch.is_tensor(input_torch)):
            input_torch = torch.from_numpy(input_torch).float()
        input_torch = input_torch.to(self.device)
        input_torch = input_torch.float()
        with torch.no_grad():
            x, mu, logvar = self.vae_net(input_torch)
            nc = self.mse(input_torch, x)
        p = (100 - stats.percentileofscore(self.nc_calibration, nc.item()))/float(100)
        return p

    def evaluate_tensor(self, input):
        input_torch = input
        input_torch = input_torch.to(self.device)
        input_torch = input_torch.float()
        with torch.no_grad():
            x, mu, logvar = self.vae_net(input_torch)
            nc = self.mse(input_torch, x)
        p = (100 - stats.percentileofscore(self.nc_calibration, nc.item()))/float(100)
        return p
