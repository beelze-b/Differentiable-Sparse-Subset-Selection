import torch

import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import math

import gc
import random


from sklearn.preprocessing import MinMaxScaler

log_interval = 20

# rounding up lowest float32 on my system
EPSILON = 1e-40


def make_encoder(input_size, hidden_layer_size, z_size):

    main_enc = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.LeakyReLU()
            #nn.BatchNorm1d(1*hidden_layer_size),
        )

    enc_mean = nn.Linear(hidden_layer_size, z_size)
    enc_logvar = nn.Linear(hidden_layer_size, z_size)

    return main_enc, enc_mean, enc_logvar


def make_bernoulli_decoder(output_size, hidden_size, z_size):

    main_dec = nn.Sequential(
            nn.Linear(z_size, 1*hidden_size),
            #nn.BatchNorm1d(hidden_size),
            #nn.LeakyReLU(),
            #nn.Linear(hidden_size, 2* hidden_size),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1* hidden_size),
            nn.Linear(1*hidden_size, output_size),
            #nn.BatchNorm1d(input_size),
            nn.Sigmoid()
        )

    return main_dec


class VAE(nn.Module):
    def __init__(self, input_size, hidden_layer_size, z_size, output_size = None):
        super(VAE, self).__init__()

        if output_size is None:
            output_size = input_size

        self.encoder, self.enc_mean, self.enc_logvar = make_encoder(input_size,
                hidden_layer_size, z_size)

        self.decoder = make_bernoulli_decoder(output_size, hidden_layer_size, z_size)


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
        return mu_x, mu_latent, logvar_latent

class VAE_l1_diag(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size):
        super(VAE_l1_diag, self).__init__(input_size, hidden_layer_size , z_size)
        
        self.diag = nn.Parameter(torch.normal(torch.zeros(input_size), 
                                 torch.ones(input_size)).requires_grad_(True))
        
    def encode(self, x):
        self.selection_layer = torch.diag(self.diag)
        xprime = torch.mm(x, self.selection_layer)
        h = self.encoder(xprime)
        return self.enc_mean(h), self.enc_logvar(h)


def gumbel_keys(w, EPSILON):
    # sample some gumbels
    uniform = (1.0 - EPSILON) * torch.rand_like(w) + EPSILON
    z = torch.log(-torch.log(uniform))
    w = w + z
    return w


#equations 3 and 4 and 5
# separate true is for debugging
def continuous_topk(w, k, t, separate=False, EPSILON = EPSILON):
    khot_list = []
    onehot_approx = torch.zeros_like(w, dtype = torch.float32)
    #print('w at start after adding gumbel noise')
    #print(w)
    for i in range(k):
        ### conver the following into pytorch
        ## ORIGINAL: khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        max_mask = 1 - onehot_approx < EPSILON
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
        
        w += torch.log(khot_mask)
        
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
def sample_subset(w, k, t, separate = False, EPSILON = EPSILON):
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
    return continuous_topk(w, k, t, separate = separate, EPSILON = EPSILON)



# L1 VAE model we are loading
class VAE_Gumbel(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01):
        super(VAE_Gumbel, self).__init__(input_size, hidden_layer_size, z_size)
        
        self.k = k
        self.t = t
        
        # end with more positive to make logit debugging easier
        
        # should probably add weight clipping to these gradients because you 
        # do not want the final output (initial logits) of this to be too big or too small
        # (values between -1 and 10 for first output seem fine)
        self.weight_creator = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            # nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, input_size),
            nn.LeakyReLU()
        )
        
    def encode(self, x):
        w = self.weight_creator(x)
        self.subset_indices = sample_subset(w, self.k, self.t)
        x = x * self.subset_indices
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)


# Not Instance_Wise Gumbel
class VAE_Gumbel_NInsta(VAE_Gumbel):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, method = 'mean'):
        super(VAE_Gumbel_NInsta, self).__init__(input_size, hidden_layer_size, z_size, k, t)
        self.method = method


    def encode(self, x):
        w0 = self.weight_creator(x)

        if self.method == 'mean':
            w = w0.mean(dim = 0).view(1, -1)
        elif self.method == 'median':
            w = w0.median(dim = 0)[0].view(1, -1)
        else:
            raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

        self.subset_indices = sample_subset(w, self.k, self.t)
        x = x * self.subset_indices
        h1 = self.encoder(x)
        return self.enc_mean(h1), self.enc_logvar(h1)


# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_GlobalGate(VAE):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01):
        super(VAE_Gumbel_GlobalGate, self).__init__(input_size, hidden_layer_size, z_size)
        
        self.k = k
        self.t = t

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size), torch.ones(input_size)).view(1, -1).requires_grad_(True))

        self.burned_in = False

    def encode(self, x):

        subset_indices = sample_subset(self.logit_enc, self.k, self.t)

        x = x * subset_indices
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

    def forward(self, x):
        mu_latent, logvar_latent = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        return mu_x, mu_latent, logvar_latent 


    def set_burned_in(self):
        self.burned_in = True
        # self.t = self.t / 10

# idea of having a Non Instance Wise Gumbel that also has a state to keep consistency across batches
# probably some repetititon of code, but the issue is this class stuff, this is python 3 tho so it can be put into a good wrapper
# that doesn't duplicate code
class VAE_Gumbel_RunningState(VAE_Gumbel):
    # alpha is for  the exponential average
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01, method = 'mean', alpha = 0.9):
        super(VAE_Gumbel_RunningState, self).__init__(input_size, hidden_layer_size, z_size, k, t)
        self.method = method

        assert alpha < 1
        assert alpha > 0

        self.logit_enc = None
        self.burned_in = False
        self.alpha = alpha
        self.logits_ae = nn.Sequential(
                nn.Linear(input_size, input_size // 4),
                nn.ReLU(),
                nn.Linear(input_size // 4, input_size)
            )

    def encode(self, x):
        if self.training:
            w = self.weight_creator(x)
            w_recon = self.logits_ae(w)

            if self.method == 'mean':
                pre_enc = w.mean(dim = 0).view(1, -1)
                w_recon_enc = w_recon.mean(dim = 0).view(1, -1)
            elif self.method == 'median':
                pre_enc = w.median(dim = 0)[0].view(1, -1)
                w_recon_enc = w_recon.median(dim = 0)[0].view(1, -1)
            else:
                raise Exception("Invalid aggregation method inside batch of Non instancewise Gumbel")

            #subset_indices = sample_subset(pre_enc, self.k, self.t)
            # state_changed_loss = F.mse_loss(w, w_recon, reduction = 'sum')

            if self.logit_enc is not None:
                # repeat used here to avoid annoying warning
                # don't use pre_enc here, since loss is spread and averaged.
                # F.mse_loss(w, self.logit_enc.repeat_interleave(w.shape[0], 0), reduction = 'sum')
                state_changed_loss = F.mse_loss(w_recon_enc, self.logit_enc.detach(), reduction = 'sum')
                self.logit_enc = (self.alpha) * self.logit_enc.detach() + (1-self.alpha) * pre_enc
                # otherwise have to keep track of a lot of gradients in the past # NOTE this applies for post burn in but detatch at every encoding because we don't now
                # self.logit_enc = self.logit_enc.detach()
            else: 
                self.logit_enc = (1-self.alpha)*pre_enc
                #self.logit_enc = pre_enc.detach()
                state_changed_loss = 0
        else:
            state_changed_loss = 0

        subset_indices = sample_subset(self.logit_enc, self.k, self.t)

        x = x * subset_indices
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1), state_changed_loss

    def forward(self, x):
        mu_latent, logvar_latent, logits_loss = self.encode(x)
        z = self.reparameterize(mu_latent, logvar_latent)
        mu_x = self.decode(z)
        return mu_x, mu_latent, logvar_latent, logits_loss


    def set_burned_in(self):
        self.burned_in = True
        # self.logit_enc = self.logit_enc.detach()
        # self.t = self.t / 10

# NMSL is Not My Selection Layer
# Implementing reference paper
class ConcreteVAE_NMSL(VAE):
    def __init__(self, input_size, hidden_layer_size, z_size, k, t = 0.01):
        # k because encoder actually uses k features as its input because of how concrete VAE picks it out
        super(ConcreteVAE_NMSL, self).__init__(k, hidden_layer_size, z_size, output_size = input_size)
        
        self.k = k
        self.t = t

        self.logit_enc = nn.Parameter(torch.normal(torch.zeros(input_size*k), torch.ones(input_size*k)).view(k, -1).requires_grad_(True))

    def encode(self, x):
        w = gumbel_keys(self.logit_enc, EPSILON = torch.finfo(torch.float32).eps)
        w = torch.softmax(w/self.t, dim = -1)

        # safe here because we do not use it in computation, only reference
        self.subset_indices = w.clone().detach()

        x = x.mm(w.transpose(0, 1))
        h1 = self.encoder(x)
        # en
        return self.enc_mean(h1), self.enc_logvar(h1)

def loss_function_per_autoencoder(x, recon_x, mu_latent, logvar_latent):
    loss_rec = F.binary_cross_entropy(recon_x, x, reduction='sum')
    

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    #print(loss_rec.item(), KLD.item())
    return loss_rec + 0.1 * KLD

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
    mu_x_1, mu_latent_1, logvar_latent_1 = ae_1(x)
    mu_x_2, mu_latent_2, logvar_latent_2 = ae_2(x)
    
    loss_vae_1 = loss_function_per_autoencoder(x, mu_x_1, mu_latent_1, logvar_latent_1)
    loss_vae_2 = loss_function_per_autoencoder(x, mu_x_2, mu_latent_2, logvar_latent_2)
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
        mu_x, mu_latent, logvar_latent = model(batch_data)
        loss = loss_function_per_autoencoder(batch_data, mu_x, mu_latent, logvar_latent) 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data)/ len(df),
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
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data)/ len(df),
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
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data)/ len(df),
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
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data) / len(df),
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
        mu_x.requires_grad_(True)
        loss = loss_function_per_autoencoder(batch_data[:, :Dim], mu_x[:, :Dim], mu_latent, logvar_latent) 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data)/ len(df),
                loss.item() / len(batch_data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(df)))
    
    return gradients
    
def train_truncated_with_gradients_gumbel_state(df, model, optimizer, epoch, batch_size, Dim, logits_changed_loss_lambda, DEBUG = False):
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
        mu_x, mu_latent, logvar_latent, logits_loss = model(batch_data)
        # why clone detach here?
        # still want gradient with respect to input, but BCE gradient with respect to target is not defined
        # plus we only want to see how input affects mu_x, not the target
        loss = loss_function_per_autoencoder(batch_data[:, :Dim].clone().detach(), mu_x[:, :Dim], 
                                             mu_latent, logvar_latent) 
        
        if DEBUG:
            innn = random.randint(1, 100)
            if innn == 10:
                print("Loss " + str(loss) + "Logits Loss " + str(logits_loss))

        loss += logits_changed_loss_lambda * logits_loss

        loss.backward(retain_graph=True)

        with torch.no_grad():
            gradients += torch.sqrt(batch_data.grad ** 2).sum(dim = 0)
        # no step
        
        optimizer.zero_grad()
        # do not calculate with respect to 
        batch_data.requires_grad_(False)
        mu_x.requires_grad_(True)
        loss = loss_function_per_autoencoder(batch_data[:, :Dim], mu_x[:, :Dim], mu_latent, logvar_latent) 
        loss += logits_changed_loss_lambda * logits_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
        
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(batch_data), len(df),
                100. * i * len(batch_data)/ len(df),
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
        nn.ReLU()
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

