# model.py
# Contains neural network model classes and optimizers
# David Leung
# Wednesday 25th September - Week 9


# Import libraries
import torch.nn as nn


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, maxlength):
        super(Discriminator, self).__init__()    # call __init__ of parent class of Discriminator (which is nn.Module)
        nFeatures = maxlength*388
        nOut = 1

        self.hiddenLayer1 = nn.Sequential(  # a sequential container for modules
            nn.Linear(nFeatures, 500),  # y = xA^T + b: linear transformation
            nn.LeakyReLU(),  # leaky ReLU: paramter controls angle of negative slope
            nn.Dropout(0.5)     # dropout layer: parameter controls probability p of zeroing
        )

        self.hiddenLayer2 = nn.Sequential(
            nn.Linear(500, 250),  # same as above but different in/out size
            nn.LeakyReLU(),  # same as hiddenlayer1
            nn.Dropout(0.5)     # same as hiddenlayer1
        )

        self.out = nn.Sequential(
            nn.Linear(250, nOut),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hiddenLayer1(x)
        x = self.hiddenLayer2(x)
        x = self.out(x)
        return x


# Generator Model
class Generator(nn.Module):
    def __init__(self, maxlength):
        super(Generator, self).__init__()
        n_features = 500
        n_out = maxlength*388

        self.hiddenlayer1 = nn.Sequential(
            nn.Linear(n_features, 500),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.hiddenlayer2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.out = nn.Sequential(
            nn.Linear(500, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hiddenlayer1(x)
        x = self.hiddenlayer2(x)
        x = self.out(x)
        return x