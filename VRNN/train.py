import math
import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from dataset import FolderDataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from librosa.output import write_wav
import time
import datetime
import matplotlib.pyplot as plt 
from model import VRNN


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""



#hyperparameters
gpu = 1
x_dim = 200 # Frame Length, corresponds to 200 consecutive raw samples
h_dim = 500 # Dimensions of the state vector
z_dim = 200 # 
n_layers =  1 
n_epochs = 500
clip = 10
learning_rate = 1e-3 #= Paper!
batch_size = 64#128 #= Paper!
seed = 128 
print_every = 50
save_every = 10
test_seq_len = 300

rnn_dim = 4000 # NOT USED! BUILD!
num_k = 1#20 # NOT! DO!

dataset_root = "datasets"
dataset_name = "intervals"
results_path = "results"

global_step = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_data_loader(bs, seq_len, dataset_root, dataset_name):
    path = os.path.join(dataset_root, dataset_name)
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=bs,
            seq_len=seq_len,
            shuffle=(not eval),
            drop_last=(not eval)
        )
    return data_loader



def train(epoch, writer):
    global global_step
    train_loss = 0
    global_step = 0
    for batch_idx, data in enumerate(train_loader):

        #transforming data
        #data = Variable(data)
        #to remove eventually
        
        # e.g. [128, 1, 28, 28] -> [28, 128, 28] - 28 sliced 128 dimensional blocks, each 28 elements
        data = Variable(data.squeeze().transpose(0, 1))
    
        # normalize input data so that mean = 0 and in range[0,1]
        data = (data - data.min().data) / (data.max().data - data.min().data)
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data.to(device))
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('train/KLD loss', kld_loss, global_step)
        writer.add_scalar('train/NLL loss', nll_loss, global_step)
        writer.add_scalar('train/Overall loss', loss, global_step)

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data / batch_size,
                nll_loss.data / batch_size))

            
            
            #plt.imshow(sample.numpy())
            #plt.pause(1e-6)
        

        global_step += 1
        
        train_loss += loss.data


    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, writer):
    """uses test data to evaluate 
    likelihood of the model"""
    
    mean_kld_loss, mean_nll_loss = 0, 0
    for i, data in enumerate(test_loader):                                            
        
        #data = Variable(data)
        data = Variable(data.squeeze().transpose(0, 1)).to(device)
        data = (data - data.min().data) / (data.max().data - data.min().data)

        kld_loss, nll_loss, _, _ = model(data)
        mean_kld_loss += kld_loss.data
        mean_nll_loss += nll_loss.data

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)
    
    overall_loss = mean_kld_loss + mean_nll_loss
    
    writer.add_scalar('test/Mean KLD loss', mean_kld_loss, global_step)
    writer.add_scalar('test/Mean NLL loss', mean_nll_loss, global_step)
    writer.add_scalar('train/Mean Overall loss', overall_loss , global_step)

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))



def generate(epoch, writer):
    samples = model.sample(test_seq_len)
    samples = samples.flatten().cpu().float().numpy()
    
    norm_samples = ((samples[:] - samples[:].min()) / (0.00001 + (samples[:].max() - samples[:].min()))) * 1.9 - 0.95

    writer.add_audio('test/sound{}'.format(global_step), norm_samples, global_step, sample_rate=16000)
    
    write_wav(os.path.join(results_path, dataset_name, "sample-{:04d}.wav".format(epoch)), samples, sr=16000, norm=True)
    writer.add_scalar('test/sample average', np.mean(samples), global_step)
    writer.add_scalar('test/sample min', samples.min(), global_step)
    writer.add_scalar('test/sample max', samples.max(), global_step)
        
        
        
if __name__ == "__main__":
    
    #manual seed
    torch.manual_seed(seed)
    plt.ion()

    
    data_loader = make_data_loader(batch_size,
                                    x_dim, dataset_root, dataset_name)
    train_loader = data_loader(0, 0.9, eval=False)
    test_loader = data_loader(0.9, 1.0, eval=True)
    
    writer = SummaryWriter(os.path.join(results_path, dataset_name))
    
    model = VRNN(x_dim, h_dim, z_dim, n_layers, num_k, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        
        #training + testing
        train(epoch, writer)
        test(epoch, writer)
        generate(epoch, writer)
        
        
        #saving model
        if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
        
