#!/usr/bin/python3
import torch.optim as optim
import time
import torch
import numpy as np
import os
from shutil import copyfile
from scipy import stats
import glob
from saliency_generator_lrp import SaliencyMapGenerator


nu = 0.1


class deepLRP():
    def __init__(self, **kwargs):
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
        self.amnetpath = None
        self.networkpath = None
        self.networkmod = None
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
        self.milestones = kwargs.get('milestones', [150, 200])
        self.num_epochs = kwargs.get('num_epochs', 250)
        self.lr_vae = kwargs.get('lr', self.lr)
        self.weight_decay_vae = kwargs.get('weight_decay', self.weight_decay)
        self.amsgrad_vae = kwargs.get('amsgrad', self.amsgrad)
        self.gamma_vae = kwargs.get('gamma', self.gamma)
        self.milestones_vae = kwargs.get('milestones', self.milestones)
        self.num_epochs_vae = kwargs.get('num_epochs', self.num_epochs)

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
            #print ' came here'
            for batch_idx, (inputs, _) in enumerate(dataloader):
                for input in inputs:
                    #print ' inside loop'
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

    def load_network(self, model_path):
        import imp
        network_path = os.path.join(model_path, "am_net.py")

        if (not os.path.exists(network_path)):
            network_path = os.path.join(model_path, '..', 'am_net.py')
        if (not os.path.exists(network_path)):
            print('network_path - default')
            import am_net_vae
            self.vae_net = am_net_vae.VAE()
            self.amnetpath = os.path.abspath(am_net_vae.__file__)[:-1]
        else:
            print('network_path ', network_path)
            self.amnetpath = network_path

            mods = imp.load_source('am_net', network_path)
            if ('VAE' in dir(mods)):
                self.vae_net = mods.VAE()

        network_path = os.path.join(model_path, "network.py")

        if (not os.path.exists(network_path)):
            network_path = os.path.join(model_path, '..', 'network.py')
        if (not os.path.exists(network_path)):
            print('network_path - default')
            import model_pytorch
            self.networkpath = os.path.abspath(model_pytorch.__file__)[:-1]
        else:
            print('network_path ', network_path)
            self.networkpath = network_path

            self.networkmod = imp.load_source('network', network_path)

    def fit(self, dataloader_training, dataloader_calibration, num_outputs, norm, model_path, network_path, **kwargs):
        self.vae_net = None
        self.net = None
        self.model_path = model_path
        self.dataloader_training = dataloader_training
        self.dataloader_calibration = dataloader_calibration
        vae_model_path = os.path.join(self.model_path, "vae.pt")
        vae_model_search_path = os.path.join(self.model_path, "..")
        nc_path = os.path.join(self.model_path, "nc_calibration.npy")
        self.load_network(model_path)
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

        mse = torch.nn.MSELoss(reduction='none')
        nc_calibration = []

        network_models = []

        nfiles = []
        for root, dirs, files in os.walk(network_path):
            for file in files:
                if file.endswith('.pt'):
                    nfiles.append(os.path.join(root, file))
                    break
            if (len(nfiles)):
                break
        if (not nfiles):
            raise ValueError('Network path is not provided for AM training')

        for i in range(num_outputs):
            if (self.networkmod):
                network_model = self.networkmod.Network(lrp=i+1)
            else:
                import model_pytorch
                network_model = model_pytorch.Network(lrp=i+1)
            network_model.load_state_dict(torch.load(
                nfiles[0], map_location=self.device))
            network_model = network_model.to(self.device)
            network_model.eval()
            network_models.append(network_model)

        saliency_generator = SaliencyMapGenerator(network_models, norm)

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
                    nc = nc.cpu().data.numpy()[0]

                    # Nag
                    # Feiyang what is the size of relevance
                    y, relevance = saliency_generator(inputs)
                    #relevance = relevance.cpu().data.numpy()[0]

                    # Mod
                    # NC_Calibration initialization for each output
                    if (not nc_calibration):
                        print 'creating nc calibration'
                        for j in range(num_outputs):
                            nc_calibration.append([])

                    # Mod
                    # For each output determine nc saliency map, by multiplying with relevance[j]
                    # update self.nc_calibration[j]
                    for j in range(num_outputs):
                        r = relevance[j].cpu().data.numpy()[0]
                        nc_saliency_map = nc * r
                        nc_saliency_map = np.sum(nc_saliency_map)
                        nc_calibration[j].append(nc_saliency_map)

            # Mod:
            # Create an np array from self.nc_calibration
            # Feiyang: Will this save as a matrix because we need nc_calibration for each output
            for j in range(num_outputs):
                nc_calibration[j] = np.asarray(nc_calibration[j])

        print("Calibration list has been constructed and totally {} data".format(
            len(nc_calibration)))

        np.save(nc_path, nc_calibration)

        dstpath = os.path.join(self.model_path, "network.pt")
        copyfile(nfiles[0], dstpath)

        am_dstpath = os.path.join(self.model_path, "am_net.py")
        if (self.amnetpath != am_dstpath):
            copyfile(self.amnetpath, am_dstpath)

        network_dstpath = os.path.join(self.model_path, "network.py")
        if (self.networkpath != network_dstpath):
            copyfile(self.networkpath, network_dstpath)

    def save_model(self, path):
        pass
