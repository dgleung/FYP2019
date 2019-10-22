from lib.midiv2 import EpianoDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from lib.utils import Logger
import progressbar
import time
import torch.utils.tensorboard

class Discriminator(nn.Module):
    def __init__(self):
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


class Generator(nn.Module):
    def __init__(self):
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


def noise(size):
    '''
    Generates 1-D vector of gaussian sampled rand values
    '''
    n = torch.randn(size, 500, requires_grad=True)
    return n


"""
Begin NN
"""
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    """Move tensor(s) to a chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


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


maxlength = 500
BATCH_SIZE = 25
trainset = DataLoader(EpianoDataset(maxlength), BATCH_SIZE, shuffle=True)
train_dl = DeviceDataLoader(trainset, device)
num_batches = len(trainset)     # number of batches

discriminator = Discriminator()
to_device(discriminator, device)
generator = Generator()
to_device(generator, device)

d_lr = 0.00001
g_lr = 0.0001
d_optimizer = optim.Adam(discriminator.parameters(), lr = d_lr)
g_optimizer = optim.Adam(generator.parameters(), lr = g_lr)


loss = nn.BCELoss()
'''
Binary Cross Entropy Loss:
L = {l1,l2....lN)^T  l(i) = -w(i) [ y(i) * log(v(i)) + (1 - y) * log(1 - v(i)) ] 
mean is calculated by computing sum(L) / N
because we don't need weights, set w(i) = 1 for all i
'''


def ones_target(size):
    return torch.ones(size, 1) - (torch.rand(size, 1)*0.2)


def zeros_target(size):
    return torch.zeros(size, 1) + (torch.rand(size, 1)*0.2)


def samples_to_vectors(samples):
    return samples.view(samples.size(0), maxlength*388)


def vectors_to_samples(vectors):
    return vectors.view(vectors.size(0), 1, -1, 388)


def train_discriminator(optimizer, realdata, fakedata):
    size = realdata.size(0)
    optimizer.zero_grad()   # reset gradients
    '''
    Discriminator Loss: (1/m * sum for i=1 to i=m) (log D(x(i)) + log (1 - D(G(z(i)))))
    '''


    '''
    Train on real data (1st half of above loss equation)
    '''
    # D(x(i))
    pred_real = discriminator(realdata)
    err_real = loss(pred_real, to_device(ones_target(size), device))   # real data has 1 target
    err_real.backward()

    '''
    Train on fake data (2nd half of above loss equation)
    '''
    # D(G(z))
    pred_fake = discriminator(fakedata)
    err_fake = loss(pred_fake, to_device(zeros_target(size), device))  # fake data has 0 target
    err_fake.backward()

    '''
    Update weights with gradients
    '''
    optimizer.step()

    '''
    Return error and predictions for real and fake inputs
    '''
    return err_real + err_fake, pred_real, pred_fake


def train_generator(optimizer, fakedata):
    size = fakedata.size(0)
    optimizer.zero_grad()   # reset gradients
    '''
    Generator Loss: (1/m * sum for i=1 to i=m) (log (1 - D(G(z(i)))))
    '''

    '''
    Train on fake data
    '''
    # D(G(z))
    prediction = discriminator(fakedata)  # this fake data comes from generator outside of function

    '''
    Calculate error and backpropagate
    '''
    error = loss(prediction, to_device(ones_target(size), device))     # instead of minimizing log(1-D(G(z))), maximise log(D(gz)) for stronger gradients in early training
    error.backward()


    '''
    Update weights with gradients
    '''
    optimizer.step()
    return error


# Testing
num_test_samples = 3
test_noise = to_device(noise(num_test_samples), device)

# Create logger instance
logger = Logger(model_name='GAN_d'+str(d_lr)+'_g'+str(g_lr), data_name='EPianoDataset')
#C:\Users\Dave\PycharmProjects\DataLoading\runs


# Printing model & optimizer state_dict
print("Discriminator state_dict:")
for param_tensor in discriminator.state_dict():
    print(param_tensor, "\t", discriminator.state_dict()[param_tensor].size())

print("\nGenerator state_dict:")
for param_tensor in generator.state_dict():
    print(param_tensor, "\t", generator.state_dict()[param_tensor].size())

print("\nD_Optimizer state_dict:")
for var_name in d_optimizer.state_dict():
    print(var_name, "\t", d_optimizer.state_dict()[var_name])

print("\nG_Optimizer state_dict:")
for var_name in g_optimizer.state_dict():
    print(var_name, "\t", g_optimizer.state_dict()[var_name])
print("\n")


# Training
num_epochs = 30
for epoch in range(num_epochs):
    for batch_num, (real_batch,_) in progressbar.progressbar(enumerate(train_dl)):   # index is discarded
        N = real_batch.size(0)

        '''1. Train Discriminator'''
        real_data = samples_to_vectors(real_batch)

        # generate fake data and detach (so gradient not calculated for generator)
        fake_data = generator(to_device(noise(N), device)).detach()

        # train discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)


        '''2. Train Generator'''
        # generate fake data (no detach this time because need graidents)
        fake_data = generator(to_device(noise(N), device))

        # train generator
        g_error = train_generator(g_optimizer, fake_data)


        ''' Log batch error '''
        logger.log(d_error, g_error, d_pred_real, d_pred_fake, epoch, batch_num, num_batches)


        ''' Display progress every few batches & save model checkpoint '''
        if (batch_num) % 100 == 0:
            generator.eval()
            test_samples = vectors_to_samples(generator(test_noise))
            test_samples = test_samples.data
            generator.train()

            logger.log_images(test_samples.cpu(), num_test_samples, epoch, batch_num, num_batches)

            # Display status logs
            logger.display_status(epoch, num_epochs, batch_num, num_batches, d_error, g_error, d_pred_real, d_pred_fake)

            # Save model dictionaries (uses HEAPS OF MEMORY ~300-600MB per save)
            torch.save(discriminator.state_dict(), './saves/D_epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
            torch.save(generator.state_dict(), './saves/G_epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
            torch.save(d_optimizer.state_dict(), './saves/Dopt_epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')
            torch.save(g_optimizer.state_dict(), './saves/Gopt_epoch'+str(epoch)+'_batch'+str(batch_num)+'.pt')


        time.sleep(0.000001)