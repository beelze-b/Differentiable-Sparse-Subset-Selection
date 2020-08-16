#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE

#import math

#import gc

from utils import *

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

import seaborn as sns
import os


# In[ ]:


torch.manual_seed(0)
np.random.seed(0)


# In[ ]:


cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

device = torch.device("cuda:0" if cuda else "cpu")
#device = 'cpu'
print("Device")
print(device)


# In[ ]:


D = 30
N = 10000
z_size = 8

# really good results for vanilla VAE on synthetic data with EPOCHS set to 50, 
# but when running locally set to 10 for reasonable run times
n_epochs = 600
batch_size = 64
lr = 0.0001
b1 = 0.9
b2 = 0.999

global_t = 4
k_lab = [D//10, D//6, D//3, D//2, D]
trial_num = 5


# In[ ]:


train_data, test_data = generate_synthetic_data_with_noise(N, z_size, D)


# In[ ]:


def train_model(train_data, model):
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr, 
                                 betas = (b1,b2))
        
    for epoch in range(1, n_epochs+1):
        train(train_data, 
              model, 
              optimizer, 
              epoch, 
              batch_size)
        model.t = max(0.001, model.t * 0.99)

        
    return model


# In[ ]:


def save_model(base_path, model):
    # make directory
    if not os.path.exists(os.path.dirname(base_path)):
        try:
            os.makedirs(os.path.dirname(base_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise Exception("COULD NOT MAKE PATH")
    with open(base_path, 'wb') as PATH:
        torch.save(model.state_dict(), PATH)


# In[ ]:


#BASE_PATH = "../data/models/final_run/"
BASE_PATH = '/scratch/ns3429/sparse-subset/data/models/final_run/'


# Vanilla VAE Gumbel First

# In[ ]:


def run_vanilla_vae_gumbel():
    for trial in range(1, trial_num+1):
        for k in k_lab:
            print("VANILLA VAE GUMBEL, TRIAL={}, K={}".format(trial, k))
            additional_path = "vanilla_vae_gumbel/k_{}/model_trial_{}.pt".format(k, trial)
            model = VAE_Gumbel(2*D, 100, 20, k = k, t = global_t)
            model.to(device)
            train_model(train_data, model)
            save_model(BASE_PATH + additional_path, model)


# In[ ]:


run_vanilla_vae_gumbel()


# Batching Gumbel VAE

# In[ ]:


def run_batching_gumbel_vae():
    for trial in range(1, trial_num+1):
        for k in k_lab:
            print("BATCHING GUMBEL VAE, TRIAL={}, K={}".format(trial, k))
            additional_path = "batching_gumbel_vae/k_{}/model_trial_{}.pt".format(k, trial)
            model = VAE_Gumbel_NInsta(2*D, 100, 20, k = k, t = global_t)
            model.to(device)
            train_model(train_data, model)
            save_model(BASE_PATH + additional_path, model)


# In[ ]:


run_batching_gumbel_vae()


# Global Gate VAE

# In[ ]:


def run_globalgate_vae():
    for trial in range(1, trial_num+1):
        for k in k_lab:
            print("GLOBAL GATE VAE, TRIAL={}, K={}".format(trial, k))
            additional_path = "globalgate_vae/k_{}/model_trial_{}.pt".format(k, trial)
            model = VAE_Gumbel_GlobalGate(2*D, 100, 20, k = k, t = global_t)
            model.to(device)
            train_model(train_data, model)
            save_model(BASE_PATH + additional_path, model)


# In[ ]:


run_globalgate_vae()


# RunningState VAE

# In[ ]:


def run_runningstate_vae():
    for trial in range(1, trial_num+1):
        for k in k_lab:
            print("RUNNING STATE VAE, TRIAL={}, K={}".format(trial, k))
            additional_path = "runningstate_vae/k_{}/model_trial_{}.pt".format(k, trial)
            model = VAE_Gumbel_RunningState(2*D, 100, 20, k = k, t = global_t, alpha = 0.9)
            model.to(device)
            train_model(train_data, model)
            save_model(BASE_PATH + additional_path, model)


# In[ ]:


run_runningstate_vae()


# ConcreteVAE

# In[ ]:


def run_concrete_vae():
    for trial in range(1, trial_num+1):
        for k in k_lab:
            print("CONCRETE VAE NMSL, TRIAL={}, K={}".format(trial, k))
            additional_path = "concrete_vae_nmsl/k_{}/model_trial_{}.pt".format(k, trial)
            model = ConcreteVAE_NMSL(2*D, 100, 20, k = k, t = global_t)
            model.to(device)
            train_model(train_data, model)
            save_model(BASE_PATH + additional_path, model)


# In[ ]:


run_concrete_vae()


# In[ ]:





# In[ ]:


model = ConcreteVAE_NMSL(2*D, 100, 20, k = 15, t = global_t)
model.to(device)


# In[ ]:


model.encoder[0].weight


# In[ ]:


model.load_state_dict(torch.load(BASE_PATH+"concrete_vae_nmsl/k_15/model_trial_1.pt"))
model.eval()


# In[ ]:


print("Ran successfully")

