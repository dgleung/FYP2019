# 23rd September Week 9 Monday
# Written by David Leung
# Pytorch Manual Dataloading Tutorial

from lib.midiv2 import EpianoDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from lib.utils import Logger
import progressbar
import time
import torch.utils.tensorboard

class EPianoLSTM(nn.Module):
    def __init__(self, batch_size, n_layers, n_steps, n_inputs, hidden_dim, n_outputs):
        super(EPianoLSTM, self).__init__()

        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs

        self.lstm = nn.LSTM(self.n_inputs, self.hidden_dim, self.n_layers, bias=True)

        #self.FC = nn.Linear(self.hidden_dim, self.n_outputs)

    def init_hidden(self):
        # (n_layers, batch_size, n_neurons)
        # initialise hidden weight matrix
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)

    def forward(self, input):
        # shape of lstm_out: [n_input, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        # reshape input to [sequence_length, batch_size, input_width]
        lstm_out, self.hidden = self.lstm(input.view(self.n_steps, self.batch_size, -1))  # perform basic_rnn pass
        #out = self.FC(lstm_out)  # perfom fully connected layer

        return lstm_out  # batch_size x n_output


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


BATCH_SIZE = 10
N_LAYERS = 1
N_STEPS = 2000
N_INPUTS = 388
N_HIDDEN = 388
N_OUTPUTS = 388
N_EPOCHS = 10

trainset = DeviceDataLoader(DataLoader(EpianoDataset(N_STEPS), BATCH_SIZE, shuffle=True), device)
num_batches = len(trainset)     # number of batches

model = EPianoLSTM(BATCH_SIZE, N_LAYERS, N_STEPS, N_INPUTS, N_HIDDEN, N_OUTPUTS)
to_device(model, device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
lossF = nn.MSELoss()

# Create logger instance
logger = Logger(model_name='LSTM', data_name='EPianoDataset')

print("LSTM state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("\nOptimizer state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

"""
TRAINING
"""
for epoch in range(N_EPOCHS):
    train_running_loss = 0.0
    train_accuracy = 0.0
    iterations = 0
    model.train()   # signal pytorch model to train mode

    # TRAINING ROUND
    # i, data are just the variables assigned to enumerate[0] and [1] respectively
    # i is index from enumerate
    # data contains array data and labels
    for i, data in progressbar.progressbar(enumerate(trainset)):
        # zero the paramater gradients
        optimizer.zero_grad()

        # reset hidden states (each minibatch)
        model.hidden = model.init_hidden()

        # get inputs and labels (data[0] and [1] respectively)
        inputs, dataset_index = data


        # forward + backward + optimize
        prediction = model(inputs)

        loss = lossF(prediction.view(10, 2000, 388), inputs)
        loss.backward()

        optimizer.step()

        # training loss
        train_running_loss += loss.detach().item()  # detach creates new tensor without requires_grad=True
        #print(train_running_loss)

        # if (i%100) == 0:
        #     savedin = inputs
        #     savedout = prediction.view(10,2000,388)
        #     print(savedin)
        #     print('\n')
        #     print(savedout)
        #     print('Newline\n')

        time.sleep(0.00001)
