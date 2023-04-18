#!/usr/bin/python3
import torch.optim as optim
import time
import torch
import numpy as np
import os
from shutil import copyfile
import json

"""

This code contains implementation for
 - training the SVDD network (for out of distribution detection)
 - computig the calibration scores based on the trained network

"""

nu = 0.1


class deepSVDD():
    def __init__(self, **kwargs):
        print("Initialize the SVDD...")
        self.dataloader_training = None
        self.dataloder_validation = None
        self.dataloader_calibration = None
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.calibration_NC = None
        self.net = None
        self.mse = torch.nn.MSELoss()
        self.model_path = None
        self.netpath = ''
        self.nc_filename = ''
        self.history = {}
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
        self.num_epochs = kwargs.get('num_epochs', 100)
        
        self.lr_svdd = kwargs.get('lr_svdd', self.lr)
        self.weight_decay_svdd = kwargs.get(
            'weight_decay_svdd', self.weight_decay)
        self.amsgrad_svdd = kwargs.get('amsgrad_svdd', self.amsgrad)
        self.gamma_svdd = kwargs.get('gamma_svdd', self.gamma)
        self.milestones_svdd = kwargs.get('milestones_svdd', self.milestones)
        self.num_epochs_svdd = kwargs.get('num_epochs_svdd', self.num_epochs)

    
    def train(self, net):
        R = torch.tensor(self.R, device=self.device)
        c = torch.tensor(self.c, device=self.device) if self.c is not None else None

        net = net.to(self.device)
        dataloader = self.dataloader_training
        val_dataloader = self.dataloader_validation
        optimizer = optim.Adam(net.parameters(), lr=self.lr_svdd, weight_decay=self.weight_decay_svdd,
                               amsgrad=self.amsgrad_svdd)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones_svdd, gamma=self.gamma_svdd)

        if c is None:
            c = self.init_center_c(dataloader, net)
            print('Center c initialized.')

        loss_array = []
        val_loss_array = []

        for epoch in range(self.num_epochs_svdd):
            print(('LR is: {}'.format(float(scheduler.get_lr()[0]))))
            if epoch in self.milestones_svdd:
                print(('  LR scheduler: new learning rate is %g' %
                       float(scheduler.get_lr()[0])))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, inputs in enumerate(dataloader):
                input = inputs[0]
                input = input.to(self.device)
                input = input.float()
                optimizer.zero_grad()
                outputs = net(input)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                if self.soft_boundary:
                    scores = dist - R ** 2
                    loss = R ** 2 + \
                        (1 / self.nu) * \
                        torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                loss.backward()
                print(('loss', loss))
                if (loss):
                    loss_epoch += loss.item()
                optimizer.step()
                if (epoch >= 10):
                    R = torch.tensor(self.get_radius(
                        dist, self.nu), device=self.device)
                    # print(R)

                n_batches += 1
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            if (n_batches):
                print(('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, self.num_epochs, epoch_train_time,
                                                                          loss_epoch / n_batches)))
            else:
                print(('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, self.num_epochs, epoch_train_time,
                                                                          loss_epoch)))
            loss_array.append(loss_epoch)

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, inputs in enumerate(val_dataloader):
                input = inputs[0]
                input = input.to(self.device)
                input = input.float()
                optimizer.zero_grad()
                #print(input.shape)
                outputs = net(input)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                if self.soft_boundary:
                    scores = dist - R ** 2
                    loss = R ** 2 + \
                        (1 / self.nu) * \
                        torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                loss.backward()
                print(('val loss', loss))
                if (loss):
                    loss_epoch += loss.item()
                optimizer.step()

                n_batches += 1
            val_loss_array.append(loss_epoch)

        return net, R, c, loss_array, val_loss_array

    def init_center_c(self, dataloader, net, eps=0.1):
        n_sample = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for batch_idx, (xinputs, _) in enumerate(dataloader):
                inputs1 = xinputs
                inputs = inputs1.to(self.device)
                inputs = inputs.float()
                outputs = net(inputs)
                n_sample += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
            c /= n_sample
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        #print(' c ', c)

        return c
    
    def get_radius(self,dist, nu):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


    def load_network(self, model_path):
        ret = None
        import imp
        svdd_network_path = os.path.join(model_path, "am_net.py")
        print(('svdd_network_path', svdd_network_path))
        if not os.path.exists(svdd_network_path):
            svdd_network_path = os.path.join(model_path, '..', 'am_net.py')
            print(('svdd_network_path', svdd_network_path))
            if not os.path.exists(svdd_network_path):
                raise ValueError(
                    'No assurance monitor network definition found')
        mods = imp.load_source('am_net', svdd_network_path)
        if 'SVDDNet' in dir(mods):
            self.net = mods.SVDDNet()
        elif 'SVDD' in dir(mods):
            self.net = mods.SVDD()
        self.netpath = svdd_network_path
        
    def fit(self, dataloader_training, dataloader_validation, dataloader_calibration, model_path, **kwargs):
        self.net = None
        self.load_network(model_path)
        if (not self.net):
            raise ValueError(
                'No SVDDNet  class definition found in assurance monitor network definition')

        self.model_path = model_path
        self.dataloader_training = dataloader_training
        self.dataloader_validation = dataloader_validation
        self.dataloader_calibration = dataloader_calibration
        self.nc_filename = 'svdd_nc.npy'

        loss_array = []
        val_loss_array = []
        print("pretraining the svdd")
        self.net, self.R, self.c, loss_array, val_loss_array = self.train(
            self.net)
        self.R = self.R.cpu().data.numpy()
        self.c = self.c.cpu().data.numpy()

        dists = []
        dataloader = self.dataloader_calibration
        #dataloader = self.dataloader_validation
        print("Constructing the list of calibration scores...")
        self.calibration_NC = []
        self.net.eval()
        #with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            
            x = inputs[0]
            #print (np.array(x).shape)
            for input in x:
                #input1 = input.view(
                #    [1, input.shape[0], input.shape[1], input.shape[2]])
                #input = input1.to(self.device)
                input = input.unsqueeze(1)
                input = input.to(self.device)
                input = input.float()
                #print(input.shape)
                with torch.no_grad():                    
                    outputs = self.net(input)
                reps = outputs.cpu().data.numpy()
                dist = np.sum((reps - self.c) ** 2, axis=1)
                self.calibration_NC.append(dist)
        self.calibration_NC = np.asarray(self.calibration_NC)
            #self.calibration_NC.sort()

        self.history = {}
        if (loss_array):
            self.history['loss'] = loss_array
        if (val_loss_array):
            self.history['val_loss'] = val_loss_array
        
        return 0, 0, 0

    
    def save_model(self, path):
        svdd_model_path = os.path.join(path, "deepSVDD.pt")
        svdd_center_path = os.path.join(path, "svdd_c.npy")
        nc_path = os.path.join(path, self.nc_filename)
        am_dstpath = os.path.join(path, "am_net.py")

        if (self.netpath != am_dstpath):
            copyfile(self.netpath, am_dstpath)

        if (self.net):
            torch.save(self.net.state_dict(), svdd_model_path)
            np.save(svdd_center_path, self.c)

        np.save(nc_path, self.calibration_NC)

        if (self.history):
            import json
            json_formatted_str = json.dumps(self.history, indent=4)
            f = open(os.path.join(path, 'history.json'), 'w')
            f.write(json_formatted_str)
            f.close()



