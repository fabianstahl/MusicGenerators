import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np

from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
    def __init__(self, x_dim, z_dim, n_layers, device, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.device = device
        
        self.num_x_features = 600
        self.num_z_features = 500
        self.num_h_features = 4000
        self.num_enc_features = 500
        self.num_dec_features = 6000
        self.num_prior_features = 500
        

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, self.num_x_features),
            nn.ReLU(),
            nn.Linear(self.num_x_features, self.num_x_features),
            nn.ReLU(),
            nn.Linear(self.num_x_features, self.num_x_features),
            nn.ReLU(),
            nn.Linear(self.num_x_features, self.num_x_features),
            nn.ReLU())
        
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, self.num_z_features),
            nn.ReLU(),
            nn.Linear(self.num_z_features, self.num_z_features),
            nn.ReLU(),
            nn.Linear(self.num_z_features, self.num_z_features),
            nn.ReLU(),
            nn.Linear(self.num_z_features, self.num_z_features),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(self.num_x_features + self.num_h_features, self.num_enc_features),
            nn.ReLU(),
            nn.Linear(self.num_enc_features, self.num_enc_features),
            nn.ReLU(),
            nn.Linear(self.num_enc_features, self.num_enc_features),
            nn.ReLU(),
            nn.Linear(self.num_enc_features, self.num_enc_features),
            nn.ReLU()
            )
        self.enc_mean = nn.Linear(self.num_enc_features, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.num_enc_features, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(self.num_h_features, self.num_prior_features),
            nn.ReLU(),
            nn.Linear(self.num_prior_features, self.num_prior_features),
            nn.ReLU(),
            nn.Linear(self.num_prior_features, self.num_prior_features),
            nn.ReLU(),
            nn.Linear(self.num_prior_features, self.num_prior_features),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.num_prior_features, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.num_prior_features, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(self.num_z_features + self.num_h_features, self.num_dec_features),
            nn.ReLU(),
            nn.Linear(self.num_dec_features, self.num_dec_features),
            nn.ReLU(),
            nn.Linear(self.num_dec_features, self.num_dec_features),
            nn.ReLU(),
            nn.Linear(self.num_dec_features, self.num_dec_features),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.num_dec_features, x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Linear(self.num_dec_features, x_dim)


        #recurrence
        self.rnn = nn.GRU(self.num_x_features + self.num_z_features, self.num_h_features, n_layers, bias) #(input_size, hidden_size, n_layers, bias)


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        #nll_loss = 0
        mse_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(1), 4000)).to(self.device)
        
        for t in range(x.size(0)):
            
            phi_x_t = self.phi_x(x[t])
            
            # Encoder (Equation 9 in Paper)
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            
            # Prior (Equation 5 in Paper)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            # Random Sampling 
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            #print("z_t", z_t.shape)
            
            # Encode the sampled z (Equation 6 in Paper) 
            phi_z_t = self.phi_z(z_t)
            #print("phi z_t", phi_z_t.shape)

            # Decoder (Equation 6 in Paper)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            
            # Recurrence / Update hidden state (Equation 7 in paper)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h) # not sure, before just h
            
            #computing losses
            kld_loss += self._kld_gauss2(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #print("kld ", kld_loss.shape) 
            
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            mse_loss += self._MSELoss(x[t], dec_mean_t)
            
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, mse_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, 4000)).to(self.device)
        
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)
            
            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data
    
        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)
        
        #a = torch.log(std_2)/torch.log(std_1) + ((std_1**2 + (mean_1-mean_2)**2)/(2*std_2)**2) - 0.5
        
        #kld = torch.sum(a, dim=-1).mean()
            
        #return kld
    
    
    def _kld_gauss2(self, mean_1, std_1, mean_2, std_2):
        loss = (2*(std_1-std_2)).exp() + ((mean_1-mean_2)/std_2.exp())**2 - 2*(std_1-std_2) - 1
        loss = 0.5*loss.sum(dim=1).mean()
        return loss


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _MSELoss(self, x, x_out):
        loss = torch.nn.functional.mse_loss(x_out, x)
        return loss


    def _nll_gauss(self, mean, std, x):

        nll = torch.sum(torch.sqrt(x-mean) / 2*(std**2) + torch.log(std) + 0.5*np.log(2*np.pi), dim=-1).mean()
        
        return nll
