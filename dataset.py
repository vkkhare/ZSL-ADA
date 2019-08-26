import os
import h5py
import hdf5storage

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import torchvision.models as models_res
import matplotlib.pyplot as plt
import cv2
import math
import itertools
import datetime
import time
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

seed = 100
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def load_checkpoint(file,model,optimizer,best_prec1=None):
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(file, checkpoint['epoch']))
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(file))
        return 0

    
    
def make_variable(tensor,volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

class MeanCovDataset(data.Dataset):
    def __init__(self, mat_file,test_dataset=False,transform=None, generalized = False):
        mat = hdf5storage.loadmat(mat_file)
        self.test_bool = test_dataset
        # Loading training data: From mat format to dictionary
        # ( Train_Classes_Size, 1)
        self.trainClassLabels= mat['trainClassLabels'].astype(int)
        # ( Test_Classes_Size, 1)
        self.testClassLabels= mat['testClassLabels'].astype(int)
        # 40 for Awa
        self.train_class_dim= len( self.trainClassLabels ) 
        # 10 for Awa
        self.test_class_dim= len( self.testClassLabels )        
        # Feat is (D*N)
        self.TestData= np.array( mat['test_feat'], dtype='float32' ).T
        
        if case==4:
            self.AttributeData= np.float32(np.load('FinalWeights/cub_attributes_reed.npy'))
        else:
            self.AttributeData= np.array( mat['classAttributes'], dtype='float32' ).T # C*D 
        self.TrainData= np.array( mat['train_feat'], dtype='float32' ).T # N*D shape, thats why tranpose
        self.TrainLabels= np.array( mat['train_labels'] ) #N*1
        self.TestLabels= np.array( mat['test_labels'] )
        self.AttributeDim= np.array( mat['classAttributes']).shape[0]
        [self.FeatureDim, self.TrainSize]= np.array( mat['train_feat'] ).shape
        self.transform = transform
        
        if generalized:
            indices = np.random.choice(self.TrainSize, int(self.TrainSize/5), replace=False)

            self.TestData = np.concatenate((self.TestData,self.TrainData[indices]),axis=0)
            self.TestLabels = np.concatenate((self.TestLabels,self.TrainLabels[indices]),axis=0)
            self.TrainData = np.delete(self.TrainData, indices, 0)
            self.TrainLabels = np.delete(self.TrainLabels, indices, 0)
            self.TrainSize = len(self.TrainData)
            self.trainClassLabels= np.unique(self.TrainLabels)
            self.testClassLabels= np.unique(self.TrainLabels)
            self.train_class_dim= len( self.trainClassLabels ) 
            self.test_class_dim= len( self.testClassLabels )
            
    def __len__(self):
        if (not self.test_bool):
            return self.TrainSize
        else:
            return len(self.TestData)

    def __getitem__(self, idx):
        if( not self.test_bool):
            x_n = self.TrainData[idx,:]
            class_label = int( self.TrainLabels[idx] )
            label_index = None
        else:
            x_n = self.TestData[idx,:]
            class_label =  int( self.TestLabels[idx] )
            label_index = np.argwhere(test_dataset.testClassLabels == class_label)[0][0]
            
        class_attribute = self.AttributeData[class_label-1,:]
        sample = {'feature': x_n, 'class_label': class_label,'attribute': class_attribute,'label_index':label_index}
        if self.transform:
            sample['feature'] = self.transform(sample['feature'])
            sample['class_label'] = self.transform(sample['class_label'])
            sample['attribute'] = self.transform(sample['attribute'])
        return sample
