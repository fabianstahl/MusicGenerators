import torch
from custom_layers import PhaseShuffle, Reshape


class Generator(torch.nn.Module):
    def __init__(self, c = 1, d = 64, samples = 16384):
        super(Generator, self).__init__()
        
        if samples == 16384:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 256 * d),
                Reshape(-1, 16 * d, 16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(16 * d, 8 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(8 * d, 4 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(4 * d, 2 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(2 * d, d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(d , c, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.Tanh()
            )
        
        elif samples == 65536:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 512 * d),
                Reshape(-1, 32 * d, 16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(32 * d, 16 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(16 * d, 8 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(8 * d, 4 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(4 * d, 2 * d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(2 * d, d, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(d , c, 25, stride = 4, padding= 11, output_padding=1),
                torch.nn.Tanh()
            )
        
        
    def forward(self, noise):
        y = self.layers(noise)
        return y
        



class Discriminator(torch.nn.Module):
    def __init__(self, c = 1, d = 64, n = 2, alpha = 0.2, samples = 16384):
        super(Discriminator, self).__init__()
        
        if samples == 16384:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv1d(c, d, 25, stride = 4),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(d, 2 * d, 25, stride = 4, padding=14),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(2 * d, 4 * d, 25, stride = 4),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(4 * d, 8 * d, 25, stride = 4, padding=14),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(8 * d, 16 * d, 25, stride = 4, padding=12),
                torch.nn.LeakyReLU(alpha),
                Reshape(-1, 256 * d),
                torch.nn.Linear(256 * d, 1)
            )
        
    
        elif samples == 65536:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv1d(c, d, 25, stride = 4),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(d, 2 * d, 25, stride = 4, padding=14),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(2 * d, 4 * d, 25, stride = 4),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(4 * d, 8 * d, 25, stride = 4, padding=14),
                torch.nn.LeakyReLU(alpha), 
                PhaseShuffle(n),
                torch.nn.Conv1d(8 * d, 16 * d, 25, stride = 4, padding=12),
                torch.nn.LeakyReLU(alpha),
                torch.nn.Conv1d(16 * d, 32 * d, 25, stride = 4, padding=12),
                torch.nn.LeakyReLU(alpha),
                Reshape(-1, 512 * d),
                torch.nn.Linear(512 * d, 1)
            )
    
    def forward(self, x):
        y = self.layers(x)
        return y
        
