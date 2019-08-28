import torch
import random

class PhaseShuffle(torch.nn.Module):
    def __init__(self, n):
        super(PhaseShuffle, self).__init__()
        self.n = n
    
    def forward(self, x):
        rand_int = random.randint(-self.n, self.n)
        #print("Rotating around '{}'".format(rand_int))
        return torch.cat((x[-rand_int:], x[:-rand_int]))



class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
