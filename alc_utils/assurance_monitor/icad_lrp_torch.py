import torch
import numpy as np
from scipy import stats
import os
from saliency_generator_lrp import SaliencyMapGenerator
import imp

# Mods:
# Accepts a norm as an input. This norm is passed to the constructor of saliencymapgenerator.
# Creates a network model from Network, loads weights and passes it to the saliencymapgenerator
# Looks for nc_calibration.npy


class ICAD():
    def __init__(self, folder_path, norm, num_outputs):
        self.mse = torch.nn.MSELoss(reduction='none')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = folder_path
        self.amnet_path = os.path.join(self.model_path, "am_net.py")
        self.vae_weights_path = os.path.join(self.model_path, "vae.pt")
        self.network_path = os.path.join(self.model_path, "network.py")
        self.network_weights_path = os.path.join(self.model_path, "network.pt")
        self.nc_calibration_path = os.path.join(
            self.model_path, "nc_calibration.npy")
        self.net = None

        if (not os.path.exists(self.amnet_path)):
            print ('AM network path does not exist')

        if (not os.path.exists(self.vae_weights_path)):
            print ('Vae weights path does not exist')

        if (not os.path.exists(self.network_path)):
            print ('Network path does not exist')

        if (not os.path.exists(self.network_weights_path)):
            print ('Network weights path does not exist')

        if (not os.path.exists(self.nc_calibration_path)):
            print ('calibration score does not exist')

        if (not os.path.exists(self.amnet_path) or not os.path.exists(self.vae_weights_path) or not os.path.exists(self.network_path)
                or not os.path.exists(self.network_weights_path) or not os.path.exists(self.nc_calibration_path)):
            print(' Missing am_net/ network definition and/or weights')
            raise ValueError(
                ' Missing am_net/ network definition and/or weights')

        mods = imp.load_source('am_net', self.amnet_path)
        if ('VAE' in dir(mods)):
            self.net = mods.VAE()
            self.net = self.net.to(self.device)
            self.net.load_state_dict(torch.load(
                self.vae_weights_path, map_location=self.device))
            self.net.eval()
        else:
            raise ValueError(
                'No VAE  class definition found in assurance monitor network definition')

        network_models = []
        self.num_outputs = num_outputs
        netmods = imp.load_source('network', self.network_path)
        if (not 'Network' in dir(netmods)):
            raise ValueError(
                'No Network class definition found in lec network definition')

        for i in range(self.num_outputs):
            network_model = netmods.Network(lrp=i+1)
            network_model.load_state_dict(torch.load(
                self.network_weights_path, map_location=self.device))
            network_model = network_model.to(self.device)
            network_model.eval()
            network_models.append(network_model)

        self.saliency_generator = SaliencyMapGenerator(network_models, norm)
        self.nc_calibration = np.load(self.nc_calibration_path)

    def evaluate(self, input):
        input_torch = input
        if (not torch.is_tensor(input_torch)):
            input_torch = torch.from_numpy(input_torch).float()
        input_torch = input_torch.to(self.device)
        input_torch = input_torch.float()
        with torch.no_grad():
            x, mu, logvar = self.net(input_torch)
            nc = self.mse(input_torch, x)
            nc = nc.cpu().data.numpy()[0]

        y, relevance = self.saliency_generator(input_torch)
        p_list = []
        r_list = []

        # compute p for each output by using the appropriate relevance and self.nc_calibration
        for j in range(self.num_outputs):
            r = relevance[j].cpu().data.numpy()[0]
            nc_saliency_map = nc * r
            nc_saliency_map = np.sum(nc_saliency_map)
            p = (
                100 - stats.percentileofscore(self.nc_calibration[j], nc_saliency_map))/float(100)
            p_list.append(p)
            r_list.append(r)
        return p_list, r_list, y
