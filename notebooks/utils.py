import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


import os
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import math

import gc
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score


import umap
import seaborn as sns
import matplotlib.pyplot as plt

log_interval = 20

# rounding up lowest float32 on my system
EPSILON = 1e-40


def make_encoder(input_size, hidden_layer_size, z_size, bias = True):

    main_enc = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size, bias = bias),
            nn.BatchNorm1d(1* hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, bias = bias),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, bias = bias),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU()
            #nn.BatchNorm1d(1*hidden_layer_size),
        )

    enc_mean = nn.Linear(hidden_layer_size, z_size, bias = bias)
    enc_logvar = nn.Linear(hidden_layer_size, z_size, bias = bias)

    return main_enc, enc_mean, enc_logvar


def make_bernoulli_decoder(output_size, hidden_size, z_size, bias = True):

    main_dec = nn.Sequential(
            nn.Linear(z_size, 1*hidden_size, bias = bias),
            nn.BatchNorm1d(hidden_size),
            #nn.LeakyReLU(),
            #nn.Linear(hidden_size, 2* hidden_size),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1* hidden_size),
            nn.Linear(1*hidden_size, output_size, bias = bias),
            #nn.BatchNorm1d(input_size),
            nn.Sigmoid()
        )


    return main_dec

def make_gaussian_decoder(output_size, hidden_size, z_size, bias = True):

    main_dec = nn.Sequential(
            nn.Linear(z_size, 1*hidden_size, bias = bias),
            nn.BatchNorm1d(hidden_size),
            #nn.LeakyReLU(),
            #nn.Linear(hidden_size, 2* hidden_size),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1* hidden_size),
            nn.Linear(1*hidden_size, output_size, bias = bias),
            # just because predicting zeisel data is >= 0
            nn.LeakyReLU()
        )

    dec_logvar = nn.Sequential(
            nn.Linear(z_size, hidden_size, bias = bias),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size, bias = bias)
            )
    
    return main_dec, dec_logvar


class VAE(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, z_size, output_size = None, bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        if output_size is None:
            output_size = input_size

        self.encoder, self.enc_mean, self.enc_logvar = make_encoder(input_size,
                hidden_layer_size, z_size, bias = bias)

        self.decoder, self.dec_logvar = make_gaussian_decoder(output_size, hidden_layer_size, z_size, bias = bias)

        self.lr = lr
        self.kl_beta = kl_beta

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):    
        return self.decoder(z)
        
    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self(x)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            mu_x, logvar_x, mu_latent, logvar_latent = self(x)
            loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)



class VAE_l1_diag(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_l1_diag, self).__init__(input_size, hidden_layer_size , z_size, bias = bias)
        self.save_hyperparameters()
        
        # using .to(device) even against pytorch lightning recs 
        # because cannot instantiate normal with it
        self.diag = nn.Parameter(torch.normal(torch.zeros(input_size, device = self.device), 
                                 torch.ones(input_size, device = self.device)).requires_grad_(True))
        
    def encode(self, x):
        self.selection_layer = torch.diag(self.diag)
        xprime = torch.mm(x, self.selection_layer)
        h = self.encoder(xprime)
        return self.enc_mean(h), self.enc_logvar(h)


def gumbel_keys(w, EPSILON):
    # sample some gumbels
    uniform = (1.0 - EPSILON) * torch.rand_like(w) + EPSILON
    z = -torch.log(-torch.log(uniform))
    w = w + z
    return w


#equations 3 and 4 and 5
# separate true is for debugging
def continuous_topk(w, k, t, device, separate=False, EPSILON = EPSILON):
    khot_list = []
    onehot_approx = torch.zeros_like(w, dtype = torch.float32, device = device)
    #print('w at start after adding gumbel noise')
    #print(w)
    for i in range(k):
        ### conver the following into pytorch
        ## ORIGINAL: khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        max_mask = (1 - onehot_approx) < EPSILON
        khot_mask = 1 - onehot_approx
        khot_mask[max_mask] = EPSILON
        
        ### t has to be close enough to zero for this to lower the logit of the previously selected thing enough
        ### otherwise it might select the same thing again
        ### and the pattern repeats if the max w values were separate enough from all the others
        
        #If debugging, uncomment these print statements
        #and return as separate to see how the logits are updated
        
        # to see if the update is big enough
        #print('Log at max values / also delta w (ignore first print since nothing updating)')
        #print(torch.log(khot_mask)[::, w.argsort(descending=True)])
        
        # does not matter if this is in-place or not because gumbel_keys happens before this
        # and creates a temporary buffer so that the model weights are not edited in place
        # make not in place just to be safe for a run
        w = w + torch.log(khot_mask)
        
        #print('w')
        #print(w)
        # to track updates
        #print("max w indices")
        #print(w.argsort(descending=True))
        # to see if the update is big enough
        #print('max w values')
        #print(w[::, w.argsort(descending=True)])
        
        
        # as in the note above about t,
        # if the differences here are not separate enough
        # might get a sitaution where the same index is selected again
        # because a flat distribution here will update all logits about the same
        
        # ORIGINAL: onehot_approx = tf.nn.softmax(w / t, axis=-1)
        onehot_approx = F.softmax(w/t, dim = -1, dtype = torch.float32)
        
        # to see if this is flat or not
        #print("One hot approx")
        #print(onehot_approx)
        
        khot_list.append(onehot_approx)
    if separate:
        return torch.stack(khot_list)
    else:
        return torch.sum(torch.stack(khot_list), dim = 0) 


# separate true is for debugging
# good default value of t looks lke 0.0001
# but let the constructor of the VAE gumbel decide that
def sample_subset(w, k, t, device, separate = False, EPSILON = EPSILON):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    #print('w before gumbel noise')
    #print(w)
    assert EPSILON > 0
    w = gumbel_keys(w, EPSILON)
    return continuous_topk(w, k, t, device, separate = separate, EPSILON = EPSILON)



# L1 VAE model we are loading
class VAE_Gumbel(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel, self).__init__(input_size, hidden_layer_size, z_size, bias = bias, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.temperature_decay = temperature_decay
        
        # end with more positive to make logit debugging easier
        
        # should probably add weight clipping to these gradients because you 
        # do not want the final output (initial logits) of this to be too big or too small
        # (values between -1 and 10 for first output seem fine)
        self.weight_creator = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, input_size),
            nn.LeakyReLU()
        )
        
    def encode(self, x):
        w = self.weight_creator(x)
        self.subset_indices = sample_subset(w, self.k, self.t, device = self.device)
        x = x * self.subset_indices
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)

    def training_epoch_end(self, training_step_outputs):
        self.t = max(0.001, self.t * self.temperature_decay)

        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None


# Not Instance_Wise Gumbel
class VAE_Gumbel_NInsta(VAE_Gumbel):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, method = 'mean', bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_NInsta, self).__init__(input_size, hidden_layer_size, z_size, k=k, t=t, temperature_decay = temperature_decay, 
                bias = bias, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        self.method = method


    def encode(self, x):
        w0 = self.weight_creator(x)

        if self.method == 'mean':
            w = w0.mean(dim = 0).view(1, -1)
        elif self.method == 'median':
            w = w0.median(dim = 0)[0].view(1, -1)
        else:
            raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

        self.subset_indices = sample_subset(w, self.k, self.t, device = self.device)
        x = x * self.subset_indices
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)


# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_GlobalGate(VAE):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_GlobalGate, self).__init__(input_size, hidden_layer_size, z_size, bias = bias, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.temperature_decay = temperature_decay

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size, device = self.device), torch.ones(input_size, device = self.device)).view(1, -1).requires_grad_(True))

        self.burned_in = False

    def encode(self, x):

        subset_indices = sample_subset(self.logit_enc, self.k, self.t, device = self.device)

        x = x * subset_indices
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

    def training_epoch_end(self, training_step_outputs):
        self.t = max(0.001, self.t * self.temperature_decay)

        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)

            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, device = self.device)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_global_gate = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_global_gate


    def set_burned_in(self):
        self.burned_in = True
        # self.t = self.t / 10

# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_RunningState(VAE_Gumbel):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, method = 'mean', alpha = 0.9, bias = True, lr = 0.000001, kl_beta = 0.1):
        super(VAE_Gumbel_RunningState, self).__init__(input_size, hidden_layer_size, z_size, k = k, t = t, temperature_decay = temperature_decay,
                bias = bias, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        self.method = method

        assert alpha < 1
        assert alpha > 0

        # flat prior for the features
        # need the view because of the way we encode
        self.register_buffer('logit_enc', torch.zeros(input_size).view(1, -1))

        self.burned_in = False
        self.alpha = alpha
        
    # training_phase determined by training_step
    def encode(self, x, training_phase=False):
        if training_phase:
            w = self.weight_creator(x)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc
            
        subset_indices = sample_subset(self.logit_enc, self.k, self.t, device = self.device)

        x = x * subset_indices
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1) 

    def forward(self, x, training_phase = False):
        mu_latent, logvar_latent = self.encode(x, training_phase = training_phase)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        logvar_x = self.dec_logvar(z)

        return mu_x, logvar_x, mu_latent, logvar_latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu_x, logvar_x, mu_latent, logvar_latent = self.forward(x, training_phase = True)
        loss = loss_function_per_autoencoder(x, mu_x, logvar_x, mu_latent, logvar_latent, kl_beta = self.kl_beta) 
        self.log('train_loss', loss)
        return loss


    def top_logits(self):
        with torch.no_grad():
            w = self.logit_enc.clone().view(-1)
            top_k_logits = torch.topk(w, k = self.k, sorted = True)[1]
            enc_top_logits = torch.nn.functional.one_hot(top_k_logits, num_classes = self.hparams.input_size).sum(dim = 0)
            
            #subsets = sample_subset(w, model.k,model.t,True)
            subsets = sample_subset(w, self.k, self.t, device = self.device)
            #max_idx = torch.argmax(subsets, 1, keepdim=True)
            #one_hot = Tensor(subsets.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)
        
        return enc_top_logits, subsets

    def markers(self):
        logits = self.top_logits()
        inds_running_state = torch.argsort(logits[0], descending = True)[:self.k]

        return inds_running_state

    def set_burned_in(self):
        self.eval()
        self.burned_in = True
        # to make sure it saves
        self.logit_enc = nn.Parameter(self.logit_enc, requires_grad = False)
        # self.logit_enc = self.logit_enc.detach()
        # self.t = self.t / 10

# NMSL is Not My Selection Layer
# Implementing reference paper
class ConcreteVAE_NMSL(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, temperature_decay = 0.99, bias = True, lr = 0.000001, kl_beta = 0.1):
        # k because encoder actually uses k features as its input because of how concrete VAE picks it out
        super(ConcreteVAE_NMSL, self).__init__(k, hidden_layer_size, z_size, output_size = input_size, bias = bias, lr = lr, kl_beta = kl_beta)
        self.save_hyperparameters()
        
        self.k = k
        self.register_buffer('t', torch.as_tensor(1.0 * t))
        self.temperature_decay = temperature_decay

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size*k, device = self.device), torch.ones(input_size*k, device = self.device)).view(k, -1).requires_grad_(True))

    def encode(self, x):
        w = gumbel_keys(self.logit_enc, EPSILON = torch.finfo(torch.float32).eps)
        w = torch.softmax(w/self.t, dim = -1)

        # safe here because we do not use it in computation, only reference
        self.subset_indices = w.clone().detach()

        x = x.mm(w.transpose(0, 1))
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

    def training_epoch_end(self, training_step_outputs):
        self.t = max(0.001, self.t * self.temperature_decay)

        loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("epoch_avg_train_loss", loss)
        return None

    def top_logits(self):
        with torch.no_grad():

            w = gumbel_keys(self.logit_enc, EPSILON = torch.finfo(torch.float32).eps)
            w = torch.softmax(w/self.t, dim = -1)
            subset_indices = w.clone().detach()

            #max_idx = torch.argmax(subset_indices, 1, keepdim=True)
            #one_hot = Tensor(subset_indices.shape)
            #one_hot.zero_()
            #one_hot.scatter_(1, max_idx, 1)

            all_subsets = subset_indices.sum(dim = 0)

            inds = torch.argsort(subset_indices.sum(dim = 0), descending = True)[:self.k]
            all_logits = torch.nn.functional.one_hot(inds, num_classes = self.hparams.input_size).sum(dim = 0)
        
        return all_logits, all_subsets

    def markers(self):
        logits = self.top_logits()
        inds_concrete = torch.argsort(logits[1], descending = True)[:self.k]

        return inds_concrete

def loss_function_per_autoencoder(x, recon_x, logvar_x, mu_latent, logvar_latent, kl_beta = 0.1):
    # loss_rec = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # loss_rec = F.mse_loss(recon_x, x, reduction='sum')
    batch_size = x.size()[0]
    loss_rec = -torch.sum(
            (-0.5 * np.log(2.0 * np.pi))
            + (-0.5 * logvar_x)
            + ((-0.5 / torch.exp(logvar_x)) * (x - recon_x) ** 2.0)
            )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    loss = (loss_rec + kl_beta * KLD) / batch_size

    return loss

# KLD of D(P_1||P_2) where P_i are Gaussians, assuming diagonal
def kld_joint_autoencoders(mu_1, mu_2, logvar_1, logvar_2):
    # equation 6 of Tutorial on Variational Autoencoders by Carl Doersch
    # https://arxiv.org/pdf/1606.05908.pdf
    mu_12 = mu_1 - mu_2
    kld = 0.5 * (-1 - (logvar_1 - logvar_2) + mu_12.pow(2) / logvar_2.exp() + torch.exp(logvar_1 - logvar_2))
    #print(kld.shape)
    kld = torch.sum(kld, dim = 1)
    
    return kld.sum()

# for joint
def loss_function_joint(x, ae_1, ae_2):
    # assuming that both autoencoders return recon_x, mu, and logvar
    # try to make ae_1 the vanilla vae
    # ae_2 should be the L1 penalty VAE
    mu_x_1, logvar_x_1, mu_latent_1, logvar_latent_1 = ae_1(x)
    mu_x_2, logvar_x_2, mu_latent_2, logvar_latent_2 = ae_2(x)
    
    loss_vae_1 = loss_function_per_autoencoder(x, mu_x_1, logvar_x_1, mu_latent_1, logvar_latent_1)
    loss_vae_2 = loss_function_per_autoencoder(x, mu_x_2, logvar_x_2, mu_latent_2, logvar_latent_2)
    joint_kld_loss = kld_joint_autoencoders(mu_latent_1, mu_latent_2, logvar_latent_1, logvar_latent_1)
    #print("Losses")
    #print(loss_vae_1)
    #print(loss_vae_2)
    #print(joint_kld_loss)
    return loss_vae_1, loss_vae_2, joint_kld_loss

def train(df, model, optimizer, epoch, batch_size):
    model.train()
    train_loss = 0
    permutations = torch.randperm(df.shape[0])
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :]
        
        optimizer.zero_grad()
        mu_x, logvar_x, mu_latent, logvar_latent = model(batch_data)
        loss = loss_function_per_autoencoder(batch_data, mu_x, logvar_x, mu_latent, logvar_latent) 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data)/ len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))
    
# match pre trained model
def train_pre_trained(df, model, optimizer, epoch, pretrained_model, batch_size):
    model.train()
    train_loss = 0
    permutations = torch.randperm(df.shape[0])
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :]
        
        optimizer.zero_grad()
        mu_x, mu_latent, logvar_latent = model(batch_data)
        with torch.no_grad():
            _, mu_latent_2, logvar_latent_2 = pretrained_model(batch_data)
        
        loss = loss_function_per_autoencoder(batch_data, mu_x, mu_latent, logvar_latent)
        loss += 10*F.mse_loss(mu_latent, mu_latent_2, reduction = 'sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data)/ len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))

# joint train two autoencoders
# model 1 should be vanilla
def train_joint(df, model1, model2, optimizer, epoch, batch_size):
    model1.train()
    model2.train()
    train_loss = 0
    permutations = torch.randperm(df.shape[0])
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :]
        
        optimizer.zero_grad()

        
        loss_vae_1, loss_vae_2, joint_kld_loss = loss_function_joint(batch_data, model1, model2)
        loss = (loss_vae_1 + loss_vae_2 + 10 * joint_kld_loss)
        loss.backward()
        
        train_loss += loss.item()
        
        optimizer.step()
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data)/ len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))


def train_l1(df, model, optimizer, epoch, batch_size):
    model.train()
    train_loss = 0
    permutations = torch.randperm(df.shape[0])
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :]
        
        optimizer.zero_grad()
        mu_x, mu_latent, logvar_latent = model(batch_data)
        loss = loss_function_per_autoencoder(batch_data, mu_x, mu_latent, logvar_latent)
        loss += 100 * torch.norm(model.diag, p = 1)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        with torch.no_grad():
            model.diag.data /= torch.norm(model.diag.data, p = 2)
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data) / len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))
    
def train_truncated_with_gradients(df, model, optimizer, epoch, batch_size, Dim):
    model.train()
    train_loss = 0
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")

    permutations = torch.randperm(df.shape[0])
    gradients = torch.zeros(df.shape[1]).to(device)
    for i in range(math.ceil(len(df)/batch_size)):
        batch_ind = permutations[i * batch_size : (i+1) * batch_size]
        batch_data = df[batch_ind, :].clone().to(device)
        
        
        # need to do this twice because deriative with respect to input not implemented in BCE
        # so need to switch them up
        optimizer.zero_grad()
        batch_data.requires_grad_(True)
        mu_x, mu_latent, logvar_latent = model(batch_data)
        # why clone detach here?
        # still want gradient with respect to input, but BCE gradient with respect to target is not defined
        # plus we only want to see how input affects mu_x, not the target
        loss = loss_function_per_autoencoder(batch_data[:, :Dim].clone().detach(), mu_x[:, :Dim], 
                                             mu_latent, logvar_latent) 
        loss.backward(retain_graph=True)

        with torch.no_grad():
            gradients += torch.sqrt(batch_data.grad ** 2).sum(dim = 0)
        # no step
        
        optimizer.zero_grad()
        # do not calculate with respect to 
        batch_data.requires_grad_(False)
        loss = loss_function_per_autoencoder(batch_data[:, :Dim], mu_x[:, :Dim], mu_latent, logvar_latent) 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * len(batch_data), len(df),
                100. * (i+1) * len(batch_data)/ len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))
    
    return gradients


def test(df, model, epoch, batch_size):
    model.eval()
    test_loss = 0
    inds = np.arange(df.shape[0])
    with torch.no_grad():
        for i in range(math.ceil(len(df)/batch_size)):
            batch_ind = inds[i * batch_size : (i+1) * batch_size]
            batch_data = df[batch_ind, :]
            mu_x, mu_latent, logvar_latent = model(batch_data)
            test_loss += loss_function_per_autoencoder(batch_data, mu_x, mu_latent, logvar_latent).item()


    test_loss /= len(df)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# test jointly two auto encoders trained together
# model1 is vanilla vae
def test_joint(df, model1, model2, epoch, batch_size):
    model1.eval()
    model2.eval()
    test_loss = 0
    inds = np.arange(df.shape[0])
    with torch.no_grad():
        for i in range(math.ceil(len(df)/batch_size)):
            batch_ind = inds[i * batch_size : (i+1) * batch_size]
            batch_data = df[batch_ind, :]
            loss_vae_1, loss_vae_2, joint_kld_loss = loss_function_joint(batch_data, model1, model2)
        
            test_loss += (loss_vae_1 + loss_vae_2 + 1000 * joint_kld_loss).item()


    test_loss /= len(df)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss



def train_model(model, train_dataloader, val_dataloader, gpus, min_epochs = 50, max_epochs = 600, auto_lr = True, max_lr = 0.001, lr_explore_mode = 'exponential'):
    assert max_epochs > 50
    early_stopping_callback = EarlyStopping(monitor='val_loss', mode = 'min', patience = 5)
    trainer = pl.Trainer(gpus = gpus, min_epochs = min_epochs, max_epochs = max_epochs, auto_lr_find=auto_lr, callbacks=[early_stopping_callback])
    if auto_lr:
        # for some reason plural val_dataloaders
        lr_finder = trainer.tuner.lr_find(model, train_dataloader = train_dataloader, val_dataloaders = val_dataloader, max_lr = max_lr, mode = lr_explore_mode)
    
    
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        print("New Learning Rate {}".format(new_lr))
    
        # update hparams of the model
        model.hparams.lr = new_lr
        model.lr = new_lr

    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer

def save_model(trainer, base_path):
    # make directory
    if not os.path.exists(os.path.dirname(base_path)):
        try:
            os.makedirs(os.path.dirname(base_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise Exception("COULD NOT MAKE PATH")
    trainer.save_checkpoint(base_path, weights_only = True)


def train_save_model(model, train_data, val_data, base_path, gpus, min_epochs, max_epochs, auto_lr = True, max_lr = 0.001, lr_explore_mode = 'exponential'):
    trainer = train_model(model, train_data, val_data, gpus, min_epochs = min_epochs, max_epochs = max_epochs, auto_lr = auto_lr, max_lr = max_lr, lr_explore_mode = lr_explore_mode)
    save_model(trainer, base_path)

def load_model(module_class, checkpoint_path):
    model = module_class.load_from_checkpoint(checkpoint_path)
    return model


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



####### Wrappyer for pytorch lightining stuff



####### Metrics

def average_cosine_angle(d1, d2):
    dotprod = torch.sum(d1*d2, dim = 1)
    lengths1 = torch.norm(d1, dim = 1)
    lengths2 = torch.norm(d2, dim = 1)

    # to handle when a vector is all 0
    markers1 = lengths1 == 0
    markers2 = lengths2 == 0
    lengths1[markers1] = 1
    lengths2[markers2] =1

    return torch.mean(dotprod / (lengths1 * lengths2))


# balanced accuracy per k
# accuracy per k
# return both train and test
# with markers and without
def metrics_model(train_data, train_labels, test_data, test_labels, markers, model, k = None):
    # if model is none don't do a confusion matrix for the model with markers

    classifier_orig = RandomForestClassifier(n_jobs = -1)
    classifier_orig_markers = RandomForestClassifier(n_jobs = -1)

    classifier_orig.fit(train_data.cpu(), train_labels)
    classifier_orig_markers.fit(train_data[:,markers].cpu(), train_labels)
    

    classifier_recon = RandomForestClassifier(n_jobs = -1)
    classifier_recon_markers = RandomForestClassifier(n_jobs = -1)

    with torch.no_grad():
        train_data_recon = model(train_data)[0].cpu()
        classifier_recon.fit(train_data_recon, train_labels)
        classifier_recon_markers.fit(train_data_recon[:, markers], train_labels)


        bac_orig = balanced_accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        bac_orig_markers = balanced_accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        bac_recon = balanced_accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        bac_recon_markers = balanced_accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))

        accuracy_orig = accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        accuracy_orig_markers = accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        accuracy_recon = accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        accuracy_recon_markers = accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))


        cos_angle_no_markers = average_cosine_angle(test_data, model(test_data)[0]).item()
        cos_angle_markers = average_cosine_angle(test_data[:, markers], model(test_data)[0][:, markers]).item()


    return {'k': k, 
            'BAC Original Data': bac_orig, 'BAC Original Data Markers': bac_orig_markers, 'BAC Recon Data': bac_recon, 'BAC Recon Data Markers': bac_recon_markers,
            'AC Original Data': accuracy_orig, 'AC Original Data Markers': accuracy_orig_markers, 'AC Recon Data': accuracy_recon, 'AC Recon Data Markers': accuracy_recon_markers,
            'Cosine Angle Between Data and Reconstruction (No Markers)': cos_angle_no_markers,
            'Cosine Angle Beteween Marked Data and Marked Reconstruction Data': cos_angle_markers
            }

def confusion_matrix_orig_recon(train_data, train_labels, test_data, test_labels, markers, model):
    # if model is none don't do a confusion matrix for the model with markers
    train_labels = zeisel_label_encoder.transform(train_labels)
    test_labels = zeisel_label_encoder.transform(test_labels)

    classifier_orig = RandomForestClassifier(n_jobs = -1)
    classifier_orig_markers = RandomForestClassifier(n_jobs = -1)

    classifier_orig.fit(train_data.cpu(), train_labels)
    classifier_orig_markers.fit(train_data[:,markers].cpu(), train_labels)
    

    classifier_recon = RandomForestClassifier(n_jobs = -1)
    classifier_recon_markers = RandomForestClassifier(n_jobs = -1)

    with torch.no_grad():
        train_data_recon = model(train_data)[0].cpu()
        classifier_recon.fit(train_data_recon, train_labels)
        classifier_recon_markers.fit(train_data_recon[:, markers], train_labels)


        cm_orig = confusion_matrix(test_labels, classifier_orig.predict(test_data.cpu()))
        cm_orig_markers = confusion_matrix(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        cm_recon = confusion_matrix(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        cm_recon_markers = confusion_matrix(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))

        accuracy_orig = accuracy_score(test_labels, classifier_orig.predict(test_data.cpu()))
        accuracy_orig_markers = accuracy_score(test_labels, classifier_orig_markers.predict(test_data[:, markers].cpu()))
        accuracy_recon = accuracy_score(test_labels, classifier_recon.predict(model(test_data)[0].cpu()))
        accuracy_recon_markers = accuracy_score(test_labels, classifier_recon_markers.predict(model(test_data)[0][:,markers].cpu()))


    print("Note: Makers here are significant for the classification. Markers are used to select which features of the (possibly Reconstructed) Data go into classifier")
    print("Confusion Matrix of Original Data")
    print(cm_orig)
    print("Accuracy {}".format(accuracy_orig))


    print("Confusion Matrix of Original Data Selected by Markers.")
    print(cm_orig_markers)
    print("Accuracy {}".format(accuracy_orig_markers))

    print("Confusion Matrix of Reconstructed Data")
    print(cm_recon)
    print("Accuracy {}".format(accuracy_recon))

    print("Confusion Matrix of Reconstructed Data by Markers")
    print(cm_recon_markers)
    print("Accuracy {}".format(accuracy_recon_markers))

#######


### graph

def graph_umap_embedding(data, labels, title, encoder):
    num_classes = len(encoder.classes_)
    data = data.detach().cpu()
    embedding = umap.UMAP(n_neighbors=10, min_dist= 0.05).fit_transform(data)
    
    
    fig, ax = plt.subplots(1, figsize=(12, 8.5))
    
    plt.scatter(*embedding.T, c = encoder.transform(labels))
    plt.setp(ax, xticks=[], yticks=[])
    
    cbar = plt.colorbar(ticks=np.arange(num_classes), boundaries = np.arange(num_classes) - 0.5)
    cbar.ax.set_yticklabels(encoder.classes_)
    
    plt.title(title)

    plt.show()

###
    

def quick_model_summary(model, train_data, test_data, threshold, batch_size):
    input_size = train_data.shape[1]
    with torch.no_grad():
        train_pred = model(train_data[0:batch_size, :])[0]
        train_pred[train_pred < threshold] = 0 

        test_pred = model(test_data[0:batch_size,:])[0]
        test_pred[test_pred < threshold] = 0 
        
    print("Per Neuron Loss Train")
    print(F.binary_cross_entropy(train_pred, train_data[0:batch_size, :], reduction='mean'))
    print("Per Neuron Loss Test")
    print(F.binary_cross_entropy(test_pred, test_data[0:batch_size, :], reduction='mean'))
    
    print("# Non Sparse in Pred test")
    print(torch.sum(test_pred[0,:] != 0))
    print("# Non Sparse in Orig test")
    print(torch.sum(test_data[0,:] != 0))


def generate_synthetic_data_with_noise(N, z_size, D, D_noise = None):
    if not D_noise:
        D_noise = D

    cuda = True if torch.cuda.is_available() else False
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    device = torch.device("cuda:0" if cuda else "cpu")


    latent_data = np.random.normal(loc=0.0, scale=1.0, size=N*z_size).reshape(N, z_size)

    data_mapper = nn.Sequential(
        nn.Linear(z_size, 2 * z_size, bias=False),
        nn.Tanh(),
        nn.Linear(2 * z_size, D, bias = True),
        nn.LeakyReLU()
        ).to(device)

    data_mapper.requires_grad_(False)

    latent_data = Tensor(latent_data)
    latent_data.requires_grad_(False)

    actual_data = data_mapper(latent_data)
    noise_features = torch.empty(N * D_noise).normal_(mean=0,std=0.01).reshape(N, D_noise).to(device)
    noise_features.requires_grad_(False)


    actual_data = torch.cat([actual_data, noise_features], dim = 1)


    actual_data = actual_data.cpu().numpy()
    scaler = MinMaxScaler()
    actual_data = scaler.fit_transform(actual_data)

    actual_data = Tensor(actual_data)

    slices = np.random.permutation(np.arange(actual_data.shape[0]))
    upto = int(.8 * len(actual_data))
    
    train_data = actual_data[slices[:upto]]
    test_data = actual_data[slices[upto:]]

    return train_data, test_data

