import os
import numpy as np
import torch
import pickle


class Dataset():
    def __init__(self, root, batch_size = 1, shuffle=True, name='Dataset', samples = 16384):
        self._samples = samples
        self._name = name
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = -1
        self._root = root
        
        extensions = [".wav", ".mp3"]
        
        self._data = pickle.load(open(self._root, "rb"))
        self._indices = [x for x in range(len(self._data))]
        
        self._size = len(self._data)
        self._scramble()
        self._i = 0
    
    def _load_chunk(self, index):
        return torch.tensor(self._data[index]).unsqueeze(dim=0)
    
    
    def _scramble(self):
        if self._shuffle:
            self._indices = np.random.permutation(self._indices)
        else:
            self._indices = [x for x in range(len(self._data))]
        self._epoch += 1
        print("Starting Epoch #{}".format(self._epoch))
        
    
    def _prepaire_batch(self):
        sample_data = []
        for _ in range(self._batch_size):
            self._i += 1
            index = self._i % self._size
            
            # case new epoch
            if index == 0:
                self._scramble()
                
            sample_data.append(self._load_chunk(self._indices[index]))
            
                
        return torch.stack(sample_data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        sample = self._prepaire_batch()
        return sample
    
    def __str__(self):
        return "Dataset '{}': '{}' samples, {} shuffled".format(self._name,  len(self._data), '' if self._shuffle else 'not')
        



    

