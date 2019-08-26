from dataset import MeanCovDataset
import os
import h5py
import hdf5storage
import copy 
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models_res
import cv2
import math
import itertools
import datetime
import time
import random

config = sys.argv[1]

#config = "running_configs/job_config.yml"
with open(config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)
ymlfile.close()

BatchSize= cfg['batch_size']
num_epochs = cfg['num_epochs']
device = torch.device('cuda:1')
kwargs = {'num_workers': 4, 'pin_memory': True}
learning_rate= cfg['lr']

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MeanCovDataset(cfg['dataset'],False)
test_dataset =  MeanCovDataset(cfg['dataset'],True) 

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=BatchSize, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=BatchSize, shuffle=True, **kwargs)

# Neural Network for learning relation between attribute vectors and parameters
zslNet = registry.construct('ada',cfg['ada'],device)
awa = BaseZSL(device)
awa.load(cfg['baseZSL_checkpoint'])
global_loss = []
partial_loss = []

for epoch in range(num_epochs):
    for i, batch in enumerate(test_loader):
        D_A.train()
        real_A = batch['feature'].cuda()
        
####      Generation from awaNet
        C_all  = torch.Tensor(dt.AttributeData[dt.testClassLabels-1])[:,0,:].cuda()
        dummy_labels = torch.arange(0, len(dt.testClassLabels)).cuda()

        iterates = int(real_A.shape[0] // len(dt.testClassLabels))
        left = real_A.shape[0] % len(dt.testClassLabels)
        sample_input = C_all.repeat(iterates,1)
        labels_B = dummy_labels.repeat(iterates)

        if left != 0:
            sample_input = torch.cat((sample_input,C_all[:left,:]),0)
            labels_B = torch.cat((labels_B,dummy_labels[:left]),0)
        means,covs = awa.model(sample_input)
        
##        reparametrisation trick
        noise = torch.randn_like(real_A)
        real_B = means.detach()+ noise * covs.detach()
        
#         prepare real and fake label
        valid = make_variable(torch.ones(real_A.size(0),1).type(torch.FloatTensor))
        fake = make_variable(torch.zeros(real_B.size(0),1).type(torch.FloatTensor))
        labels_A_soft,mask = predict_labels(real_A,test_loader,epoch)
        
        loss_G,loss_cycle,loss_identity,loss_GAN = zslNet.trainGenerator(real_A,real_B,labels_A_soft,labels_B)
        for iterate_disc in range(5):
            loss_D = zslNet.trainDiscriminators(real_A,real_B,labels_A_soft,labels_B)
            
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = num_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, num_epochs,
                                                        i, len(test_loader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

    zslNet.test_cycle(real_B,labels_B)
    zslNet.test_cycle(awa.model,real_B,labels_B)

