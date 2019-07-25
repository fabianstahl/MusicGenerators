import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted

from os import listdir
from os.path import join


class FolderDataset(Dataset):

    def __init__(self, path, ratio_min=0, ratio_max=1):
        super().__init__()
        file_names = natsorted(
            [join(path, file_name) for file_name in listdir(path)]
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
        

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()
            
            #[128, 1, 28, 28]
            
            chunked_batch = batch.reshape(batch_size, -1, self.seq_len)
            
            yield chunked_batch
            
            """
            print(batch.shape)
            
            for seq_begin in range(n_samples):
                from_index = seq_begin
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len :].contiguous()

                yield (input_sequences, reset, target_sequences)

                reset = False
            """
            
    def __len__(self):
        return super().__len__()
