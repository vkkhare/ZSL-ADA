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
zslNet = registry.construct('base_zsl',cfg['base_zsl'])
global_loss = []
partial_loss = []

for epoch in range(1, num_epochs + 1):
    acc_class= {}
    count_class= {} 
    zslNet.train_zsl(train_loader, optimizer, epoch,partial_loss) 
    zslNet.test(test_loader, epoch,global_loss, acc_class, count_class )

    if epoch % 100 ==0 or global_loss[-1] > 0.70:
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': zslNet.model.state_dict(),
        'optimizer' : zslNet.optimizer.state_dict(),
        }, False,'AwaTest/awanet_Awa_'+str(epoch)+ '_' + str(global_loss[-1])+'_.pth.tar')

print('\n Max Res: Average loss: {:.8f},\n'.format(max(global_loss)))
