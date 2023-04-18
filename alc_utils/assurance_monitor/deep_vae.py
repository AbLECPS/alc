import torch.optim as optim
import time
import torch
import numpy as np
import os
from shutil import copyfile
from scipy import stats

"""

This code contains implementation
 - training the VAE network (for out of distribution detection)
 - computig the calibration scores based on the trained network

"""

nu = 0.1

class deepVAE():
    def __init__(self, **kwargs):
        print("Initialize the VAE for regression...")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_training = None
        self.dataset_calibration = None
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.calibration_NC = []
        self.vae_net = None
        self.net = None
        self.model_path = None
        self.netpath = None
        self.mse = torch.nn.MSELoss()
        self.get_params(**kwargs)

        self.netpath = ''
        self.vae_file = ''
        self.nc_filename = ''

        self.get_params(**kwargs)


    def get_params(self, **kwargs):
        self.soft_boundary = kwargs.get('soft_boundary', False)
        self.nu = kwargs.get('nu', nu)
        self.lr = kwargs.get('lr', 0.0001)
        self.weight_decay = kwargs.get('weight_decay', 0.5e-6)
        self.amsgrad = kwargs.get('amsgrad', False)
        self.gamma = kwargs.get('gamma', 0.1)
        self.milestones = kwargs.get('milestones', [250, 350])
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.lr_vae = kwargs.get('lr_vae', self.lr)
        self.weight_decay_vae = kwargs.get(
            'weight_decay_vae', self.weight_decay)
        self.amsgrad_vae = kwargs.get('amsgrad_vae', self.amsgrad)
        self.gamma_vae = kwargs.get('gamma_vae', self.gamma)
        self.milestones_vae = kwargs.get('milestones_vae', self.milestones)
        self.num_epochs_vae = kwargs.get('num_epochs_vae', self.num_epochs)

    def train(self, vae_net):
        vae_net = vae_net.to(self.device)
        dataloader = self.dataloader_training
        val_dataloader = self.dataloader_calibration
        optimizer = optim.Adam(vae_net.parameters(), lr=self.lr_vae, weight_decay=self.weight_decay_vae,
                               amsgrad=self.amsgrad_vae)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones_vae, gamma=self.gamma_vae)
        vae_net.train()
        loss_array = []
        val_loss_array = []
        for epoch in range(self.num_epochs_vae):
            print(('LR is: {}'.format(float(scheduler.get_lr()[0]))))
            if epoch in self.milestones_vae:
                print(('  LR scheduler: new learning rate is %g' %
                       float(scheduler.get_lr()[0])))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            #print ('in vae_train epoch ', epoch)
            for batch_idx, (inputs, _) in enumerate(dataloader):
                #print ('all inputs ',inputs.shape)
                inputs = inputs.float()
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs, mu, logvar = vae_net(inputs)
                reconstruction_loss = torch.sum(
                    (outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                kl_loss = 1 + logvar - mu.pow(2) - logvar.exp()
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                loss = reconstruction_loss + kl_loss
                loss = torch.mean(loss)
                loss.backward()
                #print(('loss', loss.item()))
                if (loss):
                    loss_epoch += loss.item()
                optimizer.step()
                n_batches += 1
            loss_array.append(loss_epoch/float(n_batches))

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            val_loss_epoch = 0.0
            val_n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, _) in enumerate(val_dataloader):
                inputs = inputs.float()
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs, mu, logvar = vae_net(inputs)
                reconstruction_loss = torch.sum(
                    (outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                kl_loss = 1 + logvar - mu.pow(2) - logvar.exp()
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                loss = reconstruction_loss + kl_loss
                loss = torch.mean(loss)
                loss.backward()
                #print(('loss', loss.item()))
                if (loss):
                    val_loss_epoch += loss.item()
                optimizer.step()
                val_n_batches += 1
            val_loss_array.append(val_loss_epoch/float(val_n_batches))

            #print(n_batches)
            #print(val_n_batches)

            if (n_batches):
                print(('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Val Loss: {:.8f}'.format(epoch + 1, self.num_epochs, epoch_train_time,
                                                                          loss_epoch / float(n_batches), val_loss_epoch / float(val_n_batches))))
            else:
                print(('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t Val Loss: {:.8f}'.format(epoch + 1, self.num_epochs, epoch_train_time,
                                                                          loss_epoch, val_loss_epoch)))
            

        return vae_net, loss_array, val_loss_array


    def load_network(self, model_path):
        ret = None
        import imp
        vae_network_path = os.path.join(model_path, "am_net.py")
        print(('vae_network_path', vae_network_path))
        if not os.path.exists(vae_network_path):
            vae_network_path = os.path.join(model_path, '..', 'am_net.py')
            print(('vae_network_path', vae_network_path))
            if not os.path.exists(vae_network_path):
                raise ValueError(
                    'No assurance monitor network definition found')
        mods = imp.load_source('am_net', vae_network_path)
        if 'VAENet' in dir(mods):
            self.vae_net = mods.VAENet()
        elif 'VAE' in dir(mods):
            self.vae_net = mods.VAE()

        self.netpath = vae_network_path

    def fit(self, dataloader_training,  dataloader_validation, dataloader_calibration, model_path, **kwargs):
        self.vae_net = None
        self.load_network(model_path)
        retrain = kwargs.get('retrain',False)
        if (not self.vae_net):
            raise ValueError(
                'No VAENet  class definition found in assurance monitor network definition')
        self.vae_net = self.vae_net.to(self.device)
        self.model_path = model_path
        self.dataloader_training = dataloader_training
        self.dataloader_calibration = dataloader_calibration
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        self.nc_filename = os.path.join(self.model_path, "vae_nc.npy")

        if (self.vae_net):
            parent_model = kwargs.get("parent_model","")
            cfiles = []
            if (not parent_model):
                vae_model_search_path = os.path.join(self.model_path, "..")
                history_file = ""
                for root, dirs, files in os.walk(vae_model_search_path):
                    for file in files:
                        if file.endswith('vae.pt'):
                            cfiles.append(os.path.join(root, file))
                            history_file = os.path.join(root, "history.json")
                            break
                    if (len(cfiles)):
                        break
            else:
                parent_vae = os.path.join(parent_model,"vae.pt")
                if (os.path.exists(parent_vae)):
                    cfiles.append(parent_vae)

            if (len(cfiles)):
                self.vae_file = cfiles[0]
                self.vae_net.load_state_dict(torch.load(
                    self.vae_file, map_location=self.device))
                if (not retrain):
                    self.vae_net.eval()
                    print("loaded the pretrained vae")
                    torch.save(self.vae_net.state_dict(), vae_model_path)
                    if history_file and os.path.exists(history_file):
                        import shutil
                        history_file_copy = os.path.join(self.model_path, "history.json")
                        shutil.copyfile(history_file, history_file_copy)
                        
            if (retrain or not len(cfiles)):
                print("pretraining the vae")
                self.vae_net, loss_array, val_loss_array = self.train(self.vae_net)
                print(("saving to ", vae_model_path))
                torch.save(self.vae_net.state_dict(), vae_model_path)
                print(("saved to ", vae_model_path))
            
                self.history = {}
                if (loss_array):
                    self.history['loss'] = loss_array
                if (val_loss_array):
                    self.history['val_loss'] = val_loss_array
                if (self.history):
                    import json
                    json_formatted_str = json.dumps(self.history, indent=4)
                    f = open(os.path.join(model_path, 'history.json'), 'w')
                    f.write(json_formatted_str)
                    f.close()
                

        
        dists = []
        dataloader = self.dataloader_calibration
        print("Constructing the list of calibration scores...")
        self.vae_net.eval()
        self.calibration_NC = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                inputs = inputs.float()
                with torch.no_grad():
                    x, mu, logvar = self.vae_net(inputs)
                    nc = self.mse(inputs, x)
                self.calibration_NC.append(nc.item())
            self.calibration_NC = np.asarray(self.calibration_NC)
            #self.calibration_NC.sort()
            np.save(self.nc_filename, self.calibration_NC)

        am_dstpath = os.path.join(self.model_path, "am_net.py")
        if (self.netpath != am_dstpath):
            copyfile(self.netpath, am_dstpath)
        return 0, 0, 0

    def save_model(self, path):
        pass
