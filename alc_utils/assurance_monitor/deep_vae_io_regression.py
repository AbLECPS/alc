import torch.optim as optim
import time
import torch
import numpy as np
import os
from shutil import copyfile
from scipy import stats


nu = 0.1


class deepVAE():
    def __init__(self, **kwargs):
        print("Initialize the VAE with inputs and outputs for regression...")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_training = None
        self.dataset_calibration = None
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.calibration_NC = []
        self.vae_net = None
        self.network = None
        self.model_path = None
        self.parameters = []
        self.mse = torch.nn.MSELoss()
        self.netpath = None
        self.get_params(**kwargs)

    def get_params(self, **kwargs):
        self.soft_boundary = kwargs.get('soft_boundary', False)
        self.nu = kwargs.get('nu', nu)
        self.lr = kwargs.get('lr', 0.0001)
        self.weight_decay = kwargs.get('weight_decay', 0.5e-6)
        self.amsgrad = kwargs.get('amsgrad', False)
        self.gamma = kwargs.get('gamma', 0.1)
        self.milestones = kwargs.get('milestones', [250])
        self.num_epochs = kwargs.get('num_epochs', 350)
        self.lr_vae = kwargs.get('lr_vae', self.lr)
        self.weight_decay_vae = kwargs.get(
            'weight_decay_vae', self.weight_decay)
        self.amsgrad_vae = kwargs.get('amsgrad_vae', self.amsgrad)
        self.gamma_vae = kwargs.get('gamma_vae', self.gamma)
        self.milestones_vae = kwargs.get('milestones_vae', self.milestones)
        self.num_epochs_vae = kwargs.get('num_epochs_vae', self.num_epochs)

    def vae_train(self, vae_net):
        vae_net = vae_net.to(self.device)
        dataloader = self.dataloader_training
        vae_net.train()
        #print ('parameters {0}'.format(self.parameters))
        optimizer = optim.Adam(params=self.parameters, lr=self.lr_vae,
                               weight_decay=self.weight_decay_vae, amsgrad=self.amsgrad_vae)
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
            reconstruction_loss_epoch = 0.0
            kl_loss_epoch = 0.0
            epoch_start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                inputs = inputs.float()
                optimizer.zero_grad()
                outputs, mu, logvar, pz_mu, y = vae_net(inputs)
                reconstruction_loss = torch.sum(
                    (outputs-inputs)**2, dim=tuple(range(1, outputs.dim())))
                kl_loss = 1 + logvar - (mu-pz_mu).pow(2) - logvar.exp()
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                loss = reconstruction_loss + kl_loss
                reconstruction_loss_mean = torch.mean(reconstruction_loss)
                kl_loss_mean = torch.mean(kl_loss)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                print('loss', loss)
                if (loss):
                    loss_epoch += loss.item()
                reconstruction_loss_epoch += reconstruction_loss_mean.item()
                kl_loss_epoch += kl_loss_mean.item()
                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch +
                                                                     1, self.num_epochs, epoch_train_time, loss_epoch/n_batches))

        return vae_net

    def load_network(self, model_path):
        import imp
        network_path = os.path.join(model_path, "am_net.py")
        print('network_path', network_path)
        if (not os.path.exists(network_path)):
            network_path = os.path.join(model_path, "..", "am_net.py")
        if (os.path.exists(network_path)):
            mods = imp.load_source('am_net', network_path)
            if ('Network' in dir(mods)):
                self.network = mods.Network()
            if ('VAE' in dir(mods)):
                self.vae_net = mods.VAE()
            elif ('VAENet' in dir(mods)):
                self.vae_net = mods.VAENet()
            self.netpath = network_path

        if (not os.path.exists(network_path) or not self.network or not self.vae_net):
            import am_net_vae_io_dave2 as am_net_vae_io
            self.network = am_net_vae_io.Network()
            self.vae_net = am_net_vae_io.VAE()
            self.netpath = os.path.abspath(am_net_vae_io.__file__)[:-1]

    def init_regressor_weights(self, folder='.'):
        network_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('network.pt'):
                    network_files.append(os.path.join(root, file))
                    break
            if (len(network_files)):
                break
        if (not len(network_files)):
            return
        weights_file = network_files[0]
        self.network.load_state_dict(torch.load(weights_file))
        self.vae_net.init_regressor_weights(self.network)

    def fit(self, dataloader_training, dataloader_calibration, model_path, **kwargs):
        self.vae_net = None
        self.load_network(model_path)
        if (not self.network):
            raise ValueError(
                'No Network class definition found in assurance monitor network definition')

        if (not self.vae_net):
            raise ValueError(
                'No VAENet  class definition found in assurance monitor network definition')

        self.init_regressor_weights()
        self.parameters = self.vae_net.parameters

        self.model_path = model_path
        self.dataloader_training = dataloader_training
        self.dataloader_calibration = dataloader_calibration
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        nc_path = os.path.join(self.model_path, "vae_nc.npy")

        if (self.vae_net):
            if (os.path.exists(vae_model_path)):
                self.vae_net.load_state_dict(torch.load(vae_model_path))
                self.vae_net.eval()
                print("loaded the pretrained vae")
            else:
                print("pretraining the vae")
                self.vae_net = self.vae_train(self.vae_net)
                print("saving to ", vae_model_path)
                torch.save(self.vae_net.state_dict(), vae_model_path)
                print("saved to ", vae_model_path)

        dataloader = self.dataloader_calibration
        print("Constructing the list of calibration scores...")
        self.vae_net.eval()
        self.calibration_NC = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            inputs = inputs.float()
            with torch.no_grad():
                x, mu, logvar, pz_mu, y2 = self.vae_net(inputs)
                nc = self.mse(inputs, x)
            self.calibration_NC.append(nc.item())
        self.calibration_NC = np.asarray(self.calibration_NC)
        np.save(nc_path, self.calibration_NC)

        am_dstpath = os.path.join(self.model_path, "am_net.py")
        if (self.netpath != am_dstpath):
            copyfile(self.netpath, am_dstpath)

        return 0, 0, 0

    def save_model(self, path):
        pass
