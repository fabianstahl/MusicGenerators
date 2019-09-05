import torch
import time
import os

from tensorboardX import SummaryWriter
from data import Dataset
from models import Generator, Discriminator
from librosa.output import write_wav


gpu = 1

c = 1 # channels
d = 64 # model size
b = 64 # batch size
n = 2 # phase shuffle num
alpha = 0.2 # leaky relu steapness

samples = 65536
assert samples in [16384, 65536]
sample_rate = 16000
#epochs = 100 
steps = 1000000
dis_updates_per_gen_updates = 5
lmbda = 10

lr = 1e-4
mom1 = 0.5
mom2 = 0.9

save_every = 1000
generate_every = 100
device = 'cpu' if not torch.cuda.is_available() else 'cuda:{}'.format(gpu)
print("Running on '{}'".format(device))

results_path = "results"

train_dir = "./raw/mendelsonx4.data"
#train_dir = "./raw/drums.data"

dataset_name = train_dir.split("/")[-1].replace(".data", "")


#train_dir = "./raw/drums.data"
#val_dir = "./data/val"

gen = Generator(c, d, samples).to(device)
dis = Discriminator(c, d, n, alpha, samples).to(device)





def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda):
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



def train():
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=(mom1, mom2))
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=lr, betas=(mom1, mom2))

    writer = SummaryWriter(os.path.join(results_path, dataset_name))

    train_dataset = Dataset(train_dir, batch_size=b, shuffle=True, samples=samples)
    #val_dataset = Dataset(val_dir, batch_size=b, shuffle=False, samples=samples)

    for i in range(0,steps,dis_updates_per_gen_updates):
        
        one = torch.Tensor([1]).float().to(device)
        neg_one = (one * -1).to(device)
        
        # update generator every 5 steps
        for j in range(dis_updates_per_gen_updates):
            
            for p in dis.parameters():
                p.requires_grad = True
            
            dis.zero_grad()
            
            abs_step = i + j
            
            # generate an artificial sound batch
            noise = torch.Tensor(b, 100).uniform_(-1, 1).to(device)
            noise_var = torch.autograd.Variable(noise, requires_grad=False)
            
            train_data = next(train_dataset).to(device)
            
            
            
            data_var = torch.autograd.Variable(train_data, requires_grad=False).to(device)
            
            writer.add_audio('train/Train Music', data_var.data[0], abs_step, sample_rate=sample_rate)
            
            # send both real sound and generated sound through the discriminator
            D_real = dis(data_var).mean()
            D_real.backward(neg_one)
            
            gen_sound = torch.autograd.Variable(gen(noise_var).data)
            
            D_fake = dis(gen_sound).mean()
            D_fake.backward(one)
            
            gradient_penalty = calc_gradient_penalty(dis, data_var.data, gen_sound.data, b, lmbda)
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
        noise = torch.Tensor(b, 100).uniform_(-1, 1).to(device)
        noise_var = torch.autograd.Variable(noise, requires_grad=False)
        
        gen_music = gen(noise_var)
        dis_sound = dis(gen_music).mean()
        dis_sound.backward(neg_one)
        dis_loss = - dis_sound
        optimizer_G.step()
        
        writer.add_audio('train/Generated Music Sample 1', gen_music.data[0], i, sample_rate=sample_rate)
        writer.add_audio('train/Generated Music Sample 2', gen_music.data[1], i, sample_rate=sample_rate)
        
        if i % generate_every == 0:
        
            for k in range(3):
                writer.add_audio('train/Generated Music Sample {}'.format(k), gen_music.data[k], abs_step, sample_rate=sample_rate)
                write_wav(os.path.join(results_path, dataset_name, "sample-{:04d}-{}.wav".format(abs_step, k)), gen_music.data[k].detach().cpu().numpy().T, sr=16000, norm=True)
    
        if i % save_every == 0:
            fn1 = os.path.join(results_path, dataset_name, "model-{:05d}-gen.pth".format(abs_step))
            torch.save(gen.state_dict(), fn1)
            fn2 = os.path.join(results_path, dataset_name, "model-{:05d}-dis.pth".format(abs_step))
            torch.save(dis.state_dict(), fn2)
            print('Saved model to '+fn1 + ' and ' + fn2)
            
        

    
train()

if not os.path.exists(os.path.join(results_path, dataset_name)):
    os.path.mkdirs(os.path.join(results_path, dataset_name))
