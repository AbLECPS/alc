#!/usr/bin/python3
import torch.optim as optim
import time
import torch
import numpy as np
import os
from shutil import copyfile
from lec2_network_am import SVDD, VAE, VisualBackPropNet, get_model
from scipy import stats
import glob


nu = 0.1


class deepSaliencyMap():
    def __init__(self, **kwargs):
        print("Initialize the SVDD...")
        self.dataset_training = None
        self.dataset_calibration = None
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.calibration_NC = None
        self.vae_net = None
        self.net = None
        self.use_vae = False
        self.model_path = None
        self.only_vae = False
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.get_params(**kwargs)

    def get_params(self, **kwargs):
        self.soft_boundary = kwargs.get('soft_boundary', False)
        self.nu = kwargs.get('nu', nu)
        self.lr = kwargs.get('lr', 0.0001)
        self.weight_decay = kwargs.get('weight_decay', 0.5e-6)
        self.amsgrad = kwargs.get('amsgrad', False)
        self.gamma = kwargs.get('gamma', 0.1)
        self.milestones = kwargs.get('milestones', [250, 350])
        #self.num_epochs = kwargs.get('num_epochs',450)
        self.num_epochs = kwargs.get('num_epochs', 450)
        self.lr_vae = kwargs.get('lr_vae', self.lr)
        self.weight_decay_vae = kwargs.get(
            'weight_decay_vae', self.weight_decay)
        self.amsgrad_vae = kwargs.get('amsgrad_vae', self.amsgrad)
        self.gamma_vae = kwargs.get('gamma_vae', self.gamma)
        self.milestones_vae = kwargs.get('milestones_vae', self.milestones)
        self.num_epochs_vae = kwargs.get('num_epochs_vae', self.num_epochs)

        self.lr_svdd = kwargs.get('lr_svdd', self.lr)
        self.weight_decay_svdd = kwargs.get(
            'weight_decay_svdd', self.weight_decay)
        self.amsgrad_svdd = kwargs.get('amsgrad_svdd', self.amsgrad)
        self.gamma_svdd = kwargs.get('gamma_svdd', self.gamma)
        self.milestones_svdd = kwargs.get('milestones_svdd', self.milestones)
        self.num_epochs_svdd = kwargs.get('num_epochs_svdd', self.num_epochs)

    def vae_train(self, vae_net):

        vae_net = vae_net.to(self.device)
        dataloader = self.dataloader_training
        optimizer = optim.Adam(vae_net.parameters(
        ), lr=self.lr_vae, weight_decay=self.weight_decay_vae, amsgrad=self.amsgrad_vae)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones_vae, gamma=self.gamma_vae)
        vae_net.train()
        for epoch in range(self.num_epochs_vae):
            print('LR is: {}'.format(float(scheduler.get_lr()[0])))
            if epoch in self.milestones_vae:
                print('  LR scheduler: new learning rate is %g' %
                      float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            print ' came here'
            for batch_idx, (inputs, _) in enumerate(dataloader):
                for input in inputs:
                    print ' inside loop'
                    input = input.float()
                    input = input.to(self.device)
                    optimizer.zero_grad()

                    outputs, mu, logvar = vae_net(input)
                    reconstruction_loss = torch.sum(
                        (outputs-input)**2, dim=tuple(range(1, outputs.dim())))
                    kl_loss = 1 + logvar - mu.pow(2) - logvar.exp()
                    kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                    loss = reconstruction_loss + kl_loss
                    loss = torch.mean(loss)
                    loss.backward()
                    print ('loss', loss)
                    if (loss):
                        loss_epoch += loss.item()
                    optimizer.step()

                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            if (n_batches):
                print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch +
                                                                         1, self.num_epochs, epoch_train_time, loss_epoch/n_batches))
            else:
                print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch +
                                                                         1, self.num_epochs, epoch_train_time, loss_epoch))

        return vae_net

    def fit(self, dataloader_training, dataloader_calibration, model_path, **kwargs):
        self.vae_net = None
        self.net = None
        self.model_path = model_path
        self.dataloader_training = dataloader_training
        self.dataloader_calibration = dataloader_calibration
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        vae_model_search_path = os.path.join(self.model_path, "..")
        nc_path = os.path.join(self.model_path, "nc_calibration.npy")

        self.vae_net = VAE()
        self.vae_net.to(self.device)

        cfiles = []
        for root, dirs, files in os.walk(vae_model_search_path):
            for file in files:
                if file.endswith('vae.pt'):
                    cfiles.append(os.path.join(root, file))
                    break
            if (len(cfiles)):
                break

        if (len(cfiles)):
            vae_file = cfiles[0]
            self.vae_net.load_state_dict(torch.load(
                vae_file, map_location=self.device))
            self.vae_net.eval()
            if (os.path.abspath(files[0]) != os.path.abspath(vae_model_path)):
                torch.save(self.vae_net.state_dict(), vae_model_path)
            print("loaded the pretrained vae")

        else:
            print("pretraining the vae")
            self.vae_net = self.vae_train(self.vae_net)
            print("saving to ", vae_model_path)
            torch.save(self.vae_net.state_dict(), vae_model_path)
            print("saved to ", vae_model_path)

        model_file_name = os.path.join(self.model_path, "model_weights.h5")
        network.to(self.device)
        visual_back_prop_net = VisualBackPropNet()
        visual_back_prop_net = visual_back_prop_net.cuda()
        mse = torch.nn.MSELoss(reduction='none')
        nc_calibration = []

        dataloader = self.dataloader_training
        print("Constructing the list of calibration scores...")
        with torch.no_grad():
            for batch_idx, (xinputs, _) in enumerate(dataloader):
                for inputs in xinputs:
                    inputs = inputs.to(self.device)
                    inputs = inputs.float()
                    inputs = inputs.cuda()
                    x, mu, logvar = self.vae_net(inputs)
                    nc = mse(inputs, x)
                    output, features = network(inputs)
                    saliency_map = visual_back_prop_net(
                        features, inputs.shape)[0][0]
                    saliency = saliency_map.cpu().data.numpy()
                    nc_saliency_map = nc * saliency_map
                    nc_saliency_map = nc_saliency_map.cpu().data.numpy()
                    nc_saliency_map = np.sum(nc_saliency_map)
                    nc_calibration.append(nc_saliency_map)

        np.save(nc_path, nc_calibration)
        import lec2_network_am
        netpath = os.path.abspath(lec2_network_am.__file__)
        dstpath = os.path.join(self.model_path, "am_net.py")
        copyfile(netpath[:-1], dstpath)

    def save_model(self, path):
        pass
