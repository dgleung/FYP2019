# dataset.py
# Contains dataset class
# David Leung
# Wednesday 25th September - Week 9


# Import libraries
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy


# Dataset Class
class EpianoDataset(Dataset):
    def __init__(self, max_seq_length):
        self.maxlength = max_seq_length
        self.samples = []
        self._init_dataset()

    def __len__(self):
        return len(self.samples[0][0])

    def __getitem__(self, index):
        onevent = self.samples[0][0][index]
        volume = self.samples[0][1][index]
        offevent = self.samples[0][2][index]
        timeskip = self.samples[0][3][index]
        return self.one_hot_sample(onevent, volume, offevent, timeskip), index

    def _init_dataset(self):
        with open('./dataset/t.data', 'rb') as filehandle:
            t = pickle.load(filehandle)
        with open('./dataset/on.data', 'rb') as filehandle:
            on = pickle.load(filehandle)
        with open('./dataset/off.data', 'rb') as filehandle:
            off = pickle.load(filehandle)
        with open('./dataset/v.data', 'rb') as filehandle:
            v = pickle.load(filehandle)

        self.samples.append((on, v, off, t))

    def to_one_hot(self, eye_width, values):
        eyemat = np.eye(eye_width)
        eyemat[0,0] = 0
        return eyemat[np.asarray(values)]

    def one_hot_sample(self, onevent, volume, offevent, timeskip):
        on = self.to_one_hot(128, onevent)
        v = self.to_one_hot(32, volume)
        off = self.to_one_hot(128, offevent)
        t = self.to_one_hot(100, timeskip)
        combined = np.block([on, v, off, t])
        if combined.shape[0] < self.maxlength:
            combined = np.block([[combined], [np.zeros([self.maxlength-combined.shape[0], 388])]])
        else:
            combined = combined[0:self.maxlength,:]
        combined = from_numpy(combined)
        return combined.float()

