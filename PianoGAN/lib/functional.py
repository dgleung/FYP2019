# functional.py
# Contains functionals for training operation
# David Leung
# Wednesday 25th September - Week 9


# Import libraries
from torch import rand, zeros, ones, device, randn, from_numpy, float32
from torch.cuda import is_available
from numpy.random import choice
from numpy.random import rand as randnp
from numpy import logical_and, expand_dims
from numpy import ones as onesnp
from numpy import zeros as zerosnp

# Noise generation
def noise(size):
    return randn(size, 500, requires_grad=True)  # Generates 1-D vector of gaussian sampled rand values


# Returns targets = 1 with label smoothing and occasional target flip
def ones_target(size):
    ones_vec = onesnp(size) - (randnp(size) * 0.3)  # label smoothing
    zeros_vec = zerosnp(size) + (randnp(size) * 0.3)  # label smoothing
    flips = choice((0, 1), size, p=(0.1, 0.9))  # weighted choice generator
    notflips = onesnp(size) - flips  # elementwise inverse

    ones_vec = logical_and(ones_vec,flips)*ones_vec
    zeros_vec = logical_and(zeros_vec, notflips)*zeros_vec
    return from_numpy(expand_dims(ones_vec+zeros_vec, axis=1)).type(float32)


# Returns targets = 0 with label smothing and occasional target flip
def zeros_target(size):
    ones_vec = onesnp(size) - (randnp(size) * 0.3)  # label smoothing
    zeros_vec = zerosnp(size) + (randnp(size) * 0.3)  # label smoothing
    flips = choice((0, 1), size, p=(0.1, 0.9))  # weighted choice generator
    notflips = onesnp(size) - flips  # elementwise inverse

    ones_vec = logical_and(ones_vec, notflips) * ones_vec
    zeros_vec = logical_and(zeros_vec, flips) * zeros_vec
    return from_numpy(expand_dims(ones_vec + zeros_vec, axis=1)).type(float32)


# Generator targets = 1 (because changed to maximize rather than minimise
def generator_target(size):
    return ones(size, 1)


# Convert samples to 1D vectors
def samples_to_vectors(samples, maxlength):
    return samples.view(samples.size(0), maxlength*388)


# Convert 1D Vectors to samples
def vectors_to_samples(vectors):
    return vectors.view(vectors.size(0), 1, -1, 388)


''' GPU-DEVICE '''
# Device selection
device = device('cuda' if is_available() else 'cpu')


# Move data to selected device
def to_device(data, device):
    """Move tensor(s) to a chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Device DataLoader
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of bathces"""
        return len(self.dl)