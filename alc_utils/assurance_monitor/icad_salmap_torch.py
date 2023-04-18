#!/usr/bin/python3
import os
import numpy as np
from scipy import stats
import torch
from am_net import SVDDNet


class ICAD():
    def __init__(self, folder_path, only_vae=False):
        self.model_path = folder_path
        svdd_model_path = os.path.join(self.model_path, "deepSVDD_vae.pt")
        svdd_center_path = os.path.join(self.model_path, "svdd_c_vae.npy")
        svdd_nc_path = os.path.join(self.model_path, "svdd_nc.npy")
        print('model path', self.model_path)
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        nc_calibration_path = os.path.join(
            self.model_path, "nc_calibration.npy")
        import imp
        svdd_network_path = os.path.join(self.model_path, "am_net.py")
        print('svdd_network_path', svdd_network_path)
        foo = imp.load_source('am_net', svdd_network_path)
        model_file_name = '/aa/src/iai/aa_semseg/scripts/pytorch_semseg/models/segnet_aa_side_scan_sonar/v1.4/segnet_aa_side_scan_sonar_best_model.pkl'
        self.network = foo.get_model(model_file_name)
        self.network = self.network.cuda()
        self.visual_back_prop_net = foo.VisualBackPropNet()
        self.visual_back_prop_net = self.visual_back_prop_net.cuda()
        self.vae_net = foo.VAE()
        self.vae_net = self.vae_net.cuda()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.vae_net.load_state_dict(torch.load(
            vae_model_path, map_location=self.device))
        self.vae_net.eval()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.nc_calibration = np.load(nc_calibration_path, allow_pickle=True)

    def load_network(self):
        ret = None
        import imp
        svdd_network_path = os.path.join(self.model_path, "am_net.py")
        print('svdd_network_path', svdd_network_path)
        foo = imp.load_source('am_net', svdd_network_path)
        ret = foo.SVDDNet()
        return ret

    def evaluate(self, input):
        if isinstance(input, np.ndarray):
            input_torch = input[0]
            if (input_torch is None):
                input_torch = input[1]
        #print('input shape '+str(input_torch.shape))
        if (not torch.is_tensor(input_torch)):
            input_torch = torch.from_numpy(input_torch).float()
        if (not input_torch.is_cuda):
            #print('converting to cuda')
            input_torch = input_torch.to(self.device)
        #output = self.network(input_torch)
        x, mu, logvar = self.vae_net(input_torch)
        nc = self.mse(input_torch, x)
        output, features = self.network(input_torch)
        saliency_map = self.visual_back_prop_net(
            features, input_torch.shape)[0][0]
        saliency = saliency_map.cpu().data.numpy()
        nc_saliency_map = nc * saliency_map
        nc_saliency_map = nc_saliency_map.cpu().data.numpy()
        nc_saliency_map = np.sum(nc_saliency_map)
        p = (100 - stats.percentileofscore(self.nc_calibration,
                                           nc_saliency_map))/float(100)
        #print 'p value***** '+str(p)
        return p

    def evaluate_tensor(self, input):
        if isinstance(input, np.ndarray):
            input_torch = input[0]
            if (input_torch is None):
                input_torch = input[1]
        output = self.network(input_torch)
        x, mu, logvar = self.vae_net(input_torch)
        nc = self.mse(input_torch, x)
        output, features = self.network(input_torch)
        saliency_map = self.visual_back_prop_net(
            features, input_torch.shape)[0][0]
        saliency = saliency_map.cpu().data.numpy()
        nc_saliency_map = nc * saliency_map
        nc_saliency_map = nc_saliency_map.cpu().data.numpy()
        nc_saliency_map = np.sum(nc_saliency_map)
        p = (100 - stats.percentileofscore(self.nc_calibration,
                                           nc_saliency_map))/float(100)
        return p
