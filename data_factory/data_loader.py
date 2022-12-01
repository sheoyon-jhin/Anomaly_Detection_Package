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
# from test_aug import negative_data_aug, positive_data_aug 



class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)
        # import pdb ; pdb.set_trace()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        import pdb ; pdb.set_trace()
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class PSMSegLoader_base1(object):

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        data = pd.read_csv(data_path + '/train.csv')
        # data = remove_outlier(data,data.columns[1:]) # (87651, 26)
        
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        
        
        self.scaler.fit(data)
        
        data = self.scaler.transform(data)
        
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        
        
        
        
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:,1:]
        
        
        print("train:", self.train.shape)
        
        print("test:", self.test.shape)
        
        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            #import pdb; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.train[index:index + self.win_size])
            padded_times =torch.arange(self.win_size)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            # import pdb ; pdb.set_trace()

            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    )

        
            
        elif (self.mode == 'test'):
            # mate : add window stride size!! 
            # import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class PSMSegLoader_base(object):

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        data = pd.read_csv(data_path + '/train.csv')
        # data = remove_outlier(data,data.columns[1:]) # (87651, 26)
        
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        
        
        
        # self.scaler.fit(data)
        
        # data = self.scaler.transform(data)
        
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = test_data
        # self.test = self.scaler.transform(test_data)
# 
        self.train = data
        
        
        
        
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:,1:]
        
        
        print("train:", self.train.shape)
        
        print("test:", self.test.shape)
        
        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            #import pdb; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    )
            # item_len = self.win_size
            # padded_values= np.float32(self.train[index:index + self.win_size])
            # padded_times =torch.arange(self.win_size)
            # masks = torch.ByteTensor(self.win_size).zero_()
            # masks[:item_len] = 1
            # # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            # padded_variance = torch.ones(self.win_size)
            # # import pdb ; pdb.set_trace()

            
            # return (padded_values,
            #         padded_times,
            #         padded_variance, 
            #         masks,
            #         )

        
            
        elif (self.mode == 'test'):
            # mate : add window stride size!! 
            # import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        # import pdb ; pdb.set_trace()
        self.val = self.test # 58317,55
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class PSMSegLoader_base_latent(object):
    def __init__(self, data_path, win_size, step, mode="train", iwae = 3):
        # Q) 동적으로 받아오게 
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.iwae = iwae
        
        data = pd.read_csv(data_path + '/train.csv')
        # import pdb ;pdb.set_trace()
        data = data.values[:, 1:]
        
        
        data = np.nan_to_num(data)
        
        self.scaler.fit(data)
        
        data = self.scaler.transform(data)
        
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test_ = self.scaler.transform(test_data)

        self.train = data
        
        self.val = self.test_[:int(self.test_.shape[0]/2)]    
        self.test = self.test_[int(self.test_.shape[0]/2):]
        self.val_labels = pd.read_csv(data_path + '/test_label.csv').values[:int(self.test_.shape[0]/2), 1:]
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[int(self.test_.shape[0]/2):, 1:]
        
        
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            #import pdb; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.train[index:index + self.win_size])
            padded_times =torch.arange(self.win_size)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            # import pdb ; pdb.set_trace()

            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    )

        elif (self.mode == 'val'):
        
             #item_len = self.win_size # 100 
            item_len = self.win_size * self.iwae # 300 
            #iwae = self.iwae
            padded_values= np.float32(self.val[index:index + self.win_size* self.iwae]) # 300, 25
            
            
            padded_times =torch.arange(self.win_size* self.iwae) # 300 
            padded_times = padded_times.unsqueeze(-1) # 300,1 
            masks = torch.ByteTensor(self.win_size* self.iwae).zero_() # 300 
            masks[:item_len] = 1 # 300 
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size* self.iwae) # 300 

            labels = np.float32(self.val_labels[index:index + self.win_size* self.iwae]) # 300,1
            #import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
            
        elif (self.mode == 'test'):
            # mate : add window stride size!! 
            # import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            # padded_variance = torch.ones_like(torch.Tensor(padded_values))
            padded_variance = torch.ones(self.win_size)
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class MSLSegLoaderTest(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        
        self.val = self.test # 58317,55
        self.test_labels = np.load(data_path + "/MSL_test_label.npy") 
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        
        if self.mode == "train":
            item_len = self.win_size
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
            # mate : add window stride size!! 
            #import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            padded_variance = torch.ones(self.win_size)
            
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoaderTest(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        # import pdb ; pdb.set_trace()
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        
        if self.mode == "train":
            item_len = self.win_size
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
            # mate : add window stride size!! 
            #import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            
            
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            padded_variance = torch.ones(self.win_size)
            
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        # import pdb ; pdb.set_trace()
        self.scaler.fit(data)
        
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class SMDSegLoaderTest(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        # import pdb ; pdb.set_trace()
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

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
            # mate : add window stride size!! 
            #import pdb ; pdb.set_trace()
            item_len = self.win_size
            padded_values= np.float32(self.test[index:index + self.win_size])
            padded_times =torch.arange(self.win_size)
            padded_times = padded_times.unsqueeze(-1)
            masks = torch.ByteTensor(self.win_size).zero_()
            masks[:item_len] = 1
            padded_variance = torch.ones(self.win_size)
            
            labels = np.float32(self.test_labels[index:index + self.win_size])
            # import pdb ; pdb.set_trace()
            
            return (padded_values,
                    padded_times,
                    padded_variance, 
                    masks,
                    labels,)
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


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



def get_loader_segment(data_path, batch_size, win_size= 50, step=20, mode='train', dataset='KDD',iwae = 3):

    if (dataset == 'SMD'):
        dataset = SMDSegLoaderTest(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoaderTest(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoaderTest(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader_base1(data_path, win_size, step, mode)
    elif (dataset == 'PSM_org'):
        dataset = PSMSegLoader_base(data_path, win_size, step, mode)
    elif (dataset == 'PSM_latent'):
        dataset = PSMSegLoader_base_latent(data_path, win_size, step, mode,iwae)
    #import pdb;pdb.set_trace()
    shuffle = False
    if mode == 'train':
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)        
    return data_loader


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

