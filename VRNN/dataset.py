import torch
import numpy as np
import os

from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted


class FolderDataset(Dataset):

    def __init__(self, path, ratio_min=0, ratio_max=1):
        super().__init__()
        file_names = natsorted(
            [os.path.join(path, file_name) for file_name in os.listdir(path)]
        )

        self.file_names = file_names[
            int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.file_names[index])
        return torch.from_numpy(seq)


    def __len__(self):
        return len(self.file_names)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        
        print("calculating dataset mean ... Please wait...")
        self.mean, self.std = self.get_mean_std()
        
        # cypress hill dataset
        # self.mean = -7.28141276340466e-06
        # self.std = 0.21262432634830475
        
        # intervals (single album)
        # self.mean = 4.734374215331627e-06
        # self.std = 0.2931014895439148
        
        # intervals dataset
        # self.mean = 2.3736743059998844e-06
        # self.std = 0.29631176590919495

        # mendelssohn dataset
        # self.mean = 0.0
        # self.std = 0.03936859220266342
        
        print("dataset mean: {}, dataset std: {}".format(self.mean, self.std))

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()
            
            if n_samples % self.seq_len != 0:
                forget_index = (n_samples // self.seq_len) * self.seq_len
                batch = batch[:,:forget_index]
            
            chunked_batch = batch.reshape(batch_size, -1, self.seq_len)
            
            chunked_batch = (chunked_batch  - self.mean) / self.std
            
            yield chunked_batch
            
            
            
    def __len__(self):
        return super().__len__()


    def get_mean_std(self):
        
        mean = 0
        std = 0
        
        batches = 0
        
        for batch in super().__iter__():

            mean += torch.mean(batch)
            std += torch.std(batch)
            batches += 1
        return mean / batches, std / batches
            
            
