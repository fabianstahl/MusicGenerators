import math
import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import argparse

from tensorboardX import SummaryWriter
from dataset import FolderDataset, DataLoader
from torch.autograd import Variable
from librosa.output import write_wav
from model import VRNN


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""



default_params = {
    'gpu': 1,
    'lr': 0.0003,
    'batch_size': 128,
    'x_dim': 200,
    'z_dim': 200,
    'clip': 0.5,
    'n_rnn_layers': 1,
    'epochs': 10000,
    'save_interval': 10,
    'test_seq_len': 300,
    'x_f_dim': 600,
    'z_f_dim': 500,
    'h_dim': 2000,
    'enc_dim': 500,
    'dec_dim': 600,
    'prior_dim': 500,
    'sample_rate': 16000
}



global_step = 0



def make_data_loader(bs, seq_len, path):
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=bs,
            seq_len=seq_len,
            shuffle=(not eval),
            drop_last=True#(not eval)
        )
    return data_loader



def train(epoch, writer, train_loader, params):
    global global_step
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        # e.g. [128, 1, 28, 28] -> [28, 128, 28] - 28 sliced 128 dimensional blocks, each 28 elements
        data = Variable(data.squeeze().transpose(0, 1))
    
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data.to(device))
        loss = kld_loss + nll_loss

        if loss < 5000000:
            loss.backward()
        
            #grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
            optimizer.step()
        

        writer.add_scalar('train/KLD loss', kld_loss, global_step)
        writer.add_scalar('train/NLL loss', nll_loss, global_step)
        writer.add_scalar('train/Overall loss', loss, global_step)
        
        
        #printing progress
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data / params['batch_size'],
                nll_loss.data / params['batch_size']))


        global_step += 1        
        train_loss += loss.data
        
        break

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, writer, test_loader,params):
    """uses test data to evaluate 
    likelihood of the model"""
    
    mean_kld_loss, mean_nll_loss = 0, 0
    for i, data in enumerate(test_loader):                                            
        
        #data = Variable(data)
        data = Variable(data.squeeze().transpose(0, 1)).to(device)

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



def generate(epoch, writer, params):
    model.eval()
    samples = model.sample(params['test_seq_len'])
    model.train()
    samples = samples.flatten().cpu().float().numpy()
    
    
    
    norm_samples = ((samples[:] - samples[:].min()) / (0.00001 + (samples[:].max() - samples[:].min()))) * 1.9 - 0.95

    writer.add_audio('test/sound{}'.format(global_step), norm_samples, global_step, sample_rate=params['sample_rate'])
    dataset_name = os.path.split(params['dataset'])[-1]
    write_wav(os.path.join(params['output_dir'], dataset_name, "sample-{:04d}.wav".format(epoch)), samples, sr=params['sample_rate'], norm=True)
    writer.add_scalar('test/sample average', np.mean(samples), global_step)
    writer.add_scalar('test/sample min', samples.min(), global_step)
    writer.add_scalar('test/sample max', samples.max(), global_step)
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    
    
    parser.add_argument(
        '--gpu', type=int,
        help='which GPU to use'
    )
    parser.add_argument(
        '--lr', type=float,
        help='learning rate'
    )
    parser.add_argument(
        '--clip', type=float,
        help='clip value for gradient clipping'
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='the batch size'
    )
    parser.add_argument(
        '--x_dim', type=int,
        help='number of input sample per window'
    )
    parser.add_argument(
        '--z_dim', type=int,
        help='dimension of the latent variable z'
    )
    parser.add_argument(
        '--n_rnn_layers', type=int,
        help='number of stacked RNN layers'
    )
    parser.add_argument(
        '--epochs', type=int,
        help='stop training after this amount of epochs'
    )
    parser.add_argument(
        '--save_interval', type=int,
        help='save the model in this interval'
    )
    parser.add_argument(
        '--test_seq_len', type=int,
        help='length of test samples after each epoch'
    )
    parser.add_argument(
        '--x_f_dim', type=int,
        help='number of intermediate features in phi-x'
    )
    parser.add_argument(
        '--z_f_dim', type=int,
        help='number of intermediate features in phi-z'
    )
    parser.add_argument(
        '--enc_dim', type=int,
        help='number of intermediate features in the encoder'
    )
    parser.add_argument(
        '--dec_dim', type=int,
        help='number of intermediate features in the decoder'
    )
    parser.add_argument(
        '--h_dim', type=int,
        help='dimension of state vector h'
    )
    parser.add_argument(
        '--prior_dim', type=int,
        help='number of intermediate features in the prior'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the used music'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='name of a prepaired folder with music snippets as .wav or .mp3)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='output directory for saved graphs, samples and tensorboard logs'
    )
    

    
    parser.set_defaults(**default_params)
    params = vars(parser.parse_args())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_loader = make_data_loader(params['batch_size'],
                                    params['x_dim'], params['dataset'])
    train_loader = data_loader(0, 0.9, eval=False)
    test_loader = data_loader(0.9, 1.0, eval=True)
    
    dataset_name = os.path.split(params['dataset'])[-1]
    writer = SummaryWriter(os.path.join(params['output_dir'], dataset_name))

    model = VRNN(params, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])


    for epoch in range(1, params['epochs'] + 1):
        
        #training + testing
        train(epoch, writer, train_loader, params)
        test(epoch, writer, test_loader, params)
        generate(epoch, writer, params)
        
        
        #saving model
        if epoch % params['save_interval'] == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
        
