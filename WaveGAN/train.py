import torch
import os
import argparse

from tensorboardX import SummaryWriter
from data import Dataset
from models import Generator, Discriminator
from librosa.output import write_wav


default_params = {
    'gpu': 1,
    'lr': 1e-4,
    'alpha': 0.2,
    'phaseshuffle_n': 2,
    'batch_size': 64,
    'd': 64,
    'channels': 1,
    'samples': 65536,
    'steps': 1000000,
    'dis_up_per_gen_up': 5,
    'mom1': 0.5,
    'mom2': 0.9,
    'save_interval': 1000,
    'generate_interval': 100,
    'lambda': 10,
    'sample_rate': 16000
    }


# method taken from https://github.com/jtcramer/wavegan/blob/master/wgan.py
def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, device):
    
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def train(**params):
    
    assert params['samples'] in [16384, 65536]
    
    dataset_name = params['dataset'].split("/")[-1].replace(".data", "")
    
    # set up device
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:{}'.format(params['gpu'])
    print("Running on '{}'".format(device))
    
    # set up model
    gen = Generator(params['channels'], params['d'], params['samples']).to(device)
    dis = Discriminator(params['channels'], params['d'], params['phaseshuffle_n'], params['alpha'], params['samples']).to(device)
    
    
    # set up optimizers
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=params['lr'], betas=(params['mom1'], params['mom2']))
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=params['lr'], betas=(params['mom1'], params['mom2']))

    writer = SummaryWriter(os.path.join(params['output_dir'], dataset_name))

    # load train data
    train_dataset = Dataset(params['dataset'], batch_size=params['batch_size'], shuffle=True, samples=params['samples'])


    for i in range(0, params['steps'],params['dis_up_per_gen_up']):
        
        one = torch.Tensor([1]).float().to(device)
        neg_one = (one * -1).to(device)
        
        # update generator every 5 steps
        for j in range(params['dis_up_per_gen_up']):
            
            for p in dis.parameters():
                p.requires_grad = True
            
            dis.zero_grad()
            
            abs_step = i + j
            
            # generate an artificial sound batch
            noise = torch.Tensor(params['batch_size'], 100).uniform_(-1, 1).to(device)
            noise_var = torch.autograd.Variable(noise, requires_grad=False)
            
            train_data = next(train_dataset).to(device)
            
            data_var = torch.autograd.Variable(train_data, requires_grad=False).to(device)
            
            writer.add_audio('train/Train Music', data_var.data[0], abs_step, sample_rate=params['sample_rate'])
            
            # send both real sound and generated sound through the discriminator
            D_real = dis(data_var).mean()
            D_real.backward(neg_one)
            
            gen_sound = torch.autograd.Variable(gen(noise_var).data)
            
            D_fake = dis(gen_sound).mean()
            D_fake.backward(one)
            
            gradient_penalty = calc_gradient_penalty(dis, data_var.data, gen_sound.data, params['batch_size'], params['lambda'], device)
            gradient_penalty.backward(one)
            
            # calculate discriminator loss and update weights
            dis_loss = D_fake - D_real + gradient_penalty
            gen_loss = D_real - D_fake
            optimizer_D.step()
            
            writer.add_scalar('train/Discriminator/Real Mean', D_real, abs_step) 
            writer.add_scalar('train/Discriminator/Fake Mean', D_fake, abs_step)      
            writer.add_scalar('train/Gradient Penalty', gradient_penalty, abs_step)
            writer.add_scalar('train/Generator/Data Mean', gen_sound.mean(), abs_step)
            writer.add_scalar('train/Discriminator/Loss', dis_loss, abs_step)
            writer.add_scalar('train/Generator/Loss', gen_loss, abs_step)
            print("Step {}: D-loss={}".format(i+j, dis_loss))

        

        print("Generator Update!")
        
        for p in dis.parameters():
            p.requires_grad = False
        
        gen.zero_grad()
        
        # generator update
        noise = torch.Tensor(params['batch_size'], 100).uniform_(-1, 1).to(device)
        noise_var = torch.autograd.Variable(noise, requires_grad=False)
        
        gen_music = gen(noise_var)
        dis_sound = dis(gen_music).mean()
        dis_sound.backward(neg_one)
        dis_loss = - dis_sound
        optimizer_G.step()
        
        writer.add_audio('train/Generated Music Sample 1', gen_music.data[0], i, sample_rate=params['sample_rate'])
        writer.add_audio('train/Generated Music Sample 2', gen_music.data[1], i, sample_rate=params['sample_rate'])
        
        if i % params['generate_interval'] == 0:
        
            for k in range(min(params['batch_size'], 3)):
                writer.add_audio('train/Generated Music Sample {}'.format(k), gen_music.data[k], abs_step, sample_rate=params['sample_rate'])
                write_wav(os.path.join(params['output_dir'], dataset_name, "sample-{:04d}-{}.wav".format(abs_step, k)), gen_music.data[k].detach().cpu().numpy().T, sr=params['sample_rate'], norm=True)
    
        if i % params['save_interval'] == 0:
            fn1 = os.path.join(params['output_dir'], dataset_name, "model-{:05d}-gen.pth".format(abs_step))
            torch.save(gen.state_dict(), fn1)
            fn2 = os.path.join(params['output_dir'], dataset_name, "model-{:05d}-dis.pth".format(abs_step))
            torch.save(dis.state_dict(), fn2)
            print('Saved model to '+fn1 + ' and ' + fn2)
            
        


if __name__ == '__main__':
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
        '--alpha', type=float,
        help='Leaky Relu factor'
    )
    parser.add_argument(
        '--phaseshuffle_n', type=int,
        help='maximal shuffeling offset in discriminator'
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='the batch size'
    )
    parser.add_argument(
        '--d', type=int,
        help='the models scale factor'
    )
    parser.add_argument(
        '--channels', type=int,
        help='number of audio channels'
    )
    parser.add_argument(
        '--samples', type=int,
        help='number of input / output samples, only 16384 and 65536 allowed'
    )
    parser.add_argument(
        '--steps', type=int,
        help='stop training after this amount of training steps'
    )
    parser.add_argument(
        '--dis_up_per_gen_up', type=int,
        help='discriminator updates per generator update'
    )
    parser.add_argument(
        '--mom1', type=float,
        help='Adam optimizer beta1'
    )
    parser.add_argument(
        '--mom2', type=float,
        help='Adam optimizer beta2'
    )
    parser.add_argument(
        '--save_interval', type=int,
        help='save the model in this interval'
    )
    parser.add_argument(
        '--generate_interval', type=int,
        help='generates samples in this interval'
    )
    parser.add_argument(
        '--lambda', type=int,
        help='lambda value of gradient penalty'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='the sample rate (samples per second)'
    )    
    parser.add_argument(
        '--dataset', required=True,
        help='name of a prepaired .data numpy arraw (using prepaire_Music_chunks.py))'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='output directory for saved graphs, samples and tensorboard logs'
    )

    
    parser.set_defaults(**default_params)
    
    
    train(**vars(parser.parse_args()))

if not os.path.exists(os.path.join(results_path, dataset_name)):
    os.path.mkdirs(os.path.join(results_path, dataset_name))
