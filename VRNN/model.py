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
    def __init__(self, x_dim, h_dim, z_dim, n_layers, num_k, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.num_k = num_k

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
            )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, self.num_k * x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Linear(h_dim, self.num_k * x_dim)


        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias) #(input_size, hidden_size, n_layers, bias)


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to("cuda")
        
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
            
            # Encode the sampled z (Equation 6 in Paper)
            phi_z_t = self.phi_z(z_t)

            # Decoder (Equation 6 in Paper)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)


            # Recurrence / Update hidden state (Equation 7 in paper)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h) # not sure, before just h

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            
            
            #nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
            
            #x_in = x.reshape((x.shape[0]*x.shape[1],-1))
            
            #nll_loss += self._GMM(x_in, dec_mean_t, dec_std_t, coeff)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).to("cuda")
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
        eps = Variable(eps).to("cuda")
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element, dim=-1).mean()


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))



    def _GMM(self, y, mu, sig, coeff):
        
        print(y.shape, mu.shape, sig.shape, coeff.shape)
        #y = y.dimshuffle(0, 1, 'x')
        y = torch.unsqueeze(y, dim=2)
        
        mu = mu.reshape((mu.shape[0],
                    mu.shape[1]//coeff.shape[-1],
                    coeff.shape[-1]))
        
        sig = sig.reshape((sig.shape[0],
                    sig.shape[1]//coeff.shape[-1],
                    coeff.shape[-1]))
        
        print(y.shape, mu.shape, sig.shape, coeff.shape)
        
        #inner = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
        #inner = -0.5 * np.sum(((y-mu)**2) / sig**2 + 2 * np.log(sig) + np.log(2*np.pi), axis=1)
        
        #a = T.log(coeff) + inner
        a = np.log(coeff) + inner
        
        #x_max = T.max(nll, axis=1, keepdims=True)
        x_max = np.max(a, axis=1, keepdims=True)
        
        # z = T.log(T.sum(T.exp(a - x_max), axis=1, keepdims=True)) + x_max
        z = np.log(np.sum(np.exp(a-x_max), axis=1, keepdims=True)) + x_max
        
        z = z.sum(axis=1)
        
        return -z



    def _nll_gauss(self, mean, std, x):

        nll = 0.5 * torch.sum((x-mean).pow(2) / (std**2) + 2 * torch.log(std) + np.log(2*np.pi), dim=-1).mean()
        
        return nll
