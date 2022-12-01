from hashlib import pbkdf2_hmac
import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle



class SMDSegLoaderBase(object):
    def __init__(self,train_file,test_file,test_file_label, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path +train_file)
        self.scaler.fit(data)
        
        data = self.scaler.transform(data)
        test_data = np.load(data_path + test_file)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + test_file_label)
        

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        
        if self.mode == "train":
            item_len = self.win_size
            if (len(self.test)<index+self.win_size):
                padded_values= np.float32(self.train[index:])
            else: 
                padded_values= np.float32(self.train[index:index + self.win_size])
            
            padded_times =torch.arange(self.win_size)
            
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            padded_variance = torch.ones(self.win_size)
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    )

        
            
        elif (self.mode == 'test'):
            item_len = self.win_size
            if (len(self.test)<index+self.win_size):
                padded_values= np.float32(self.test[index:])
            else: 
                padded_values= np.float32(self.test[index:index + self.win_size])
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            padded_variance = torch.ones(self.win_size)
            
            labels = np.float32(self.test_labels[index:index + self.win_size])
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])





def remove_outlier(dataframe, col):
    dff = dataframe
    for k in col:
        #import pdb; pdb.set_trace()
        level_1q = dff[k].quantile(0.25)
        level_3q = dff[k].quantile(0.75)
        IQR = level_3q - level_1q
        rev_range = 3 # 제거 범위 조절 변수
        dff = dff[(dff[k] <= level_3q + (rev_range * IQR)) & (dff[k] >= level_1q - (rev_range * IQR))]
        dff = dff.reset_index(drop=True)
    return dff

def get_loader_segment_base(data_path, batch_size, win_size= 50, step=20, mode='train', dataset='KDD',iwae = 3):

    if (dataset == 'SMD'):
        dataset = SMDSegLoaderTest(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoaderTest(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoaderTest(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader_base1(data_path, win_size, step, mode)
    
    #import pdb;pdb.set_trace()
    shuffle = False
    if mode == 'train':
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)        
    return data_loader

