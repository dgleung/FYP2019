import numpy as np
import torchvision
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class CustomDatasetFromNpy(Dataset):
    def __init__(self, img_path, labels_path, width, height, transform=None):

        # Pytorch transforms for transforms and tensor conversion
        self.transform = transform

        # Read Numpy file
        self.images_arr = np.load(img_path)
        self.labels_arr = np.load(labels_path)

        # Get Image Dimensions
        self.width = width
        self.height = height

    def __getitem__(self, item):
        # The goal in this method is to pass one instance at a time
        # The labels don't need any preprocessing
        single_image_label = self.labels_arr[item]

        # Need to change the image data to follow image convention
        single_image = self.images_arr[item].reshape(self.width, self.height).astype('uint8')
        single_image_as_image = Image.fromarray(single_image)
        single_image_as_image = single_image_as_image.convert('L')

        # Now need to apply transformations
        if self.transform is not None:
            final_img = self.transform(single_image_as_image)        

        return final_img, single_image_label

    def __len__(self):
        return self.images_arr.shape[0]


x_train_path =          './mnist/mnist_train_data.npy'
x_train_label_path =    './mnist/mnist_train_label.npy'

x_test_path =           './mnist/mnist_test_data.npy'
x_test_label_path =     './mnist/mnist_test_label.npy'

transformations = transforms.Compose([transforms.ToTensor()])
train_set = CustomDatasetFromNpy(x_train_path, x_train_label_path, 28, 28, transformations)
test_set = CustomDatasetFromNpy(x_test_path, x_test_label_path, 28, 28, transformations)

train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=100,
        shuffle=True
    )

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=100,
        shuffle=False
    )


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fcl_ih = torch.nn.Linear(input_dim, hidden_dim)
        self.fcl_ho = torch.nn.Linear(hidden_dim, output_dim)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        out = self.fcl_ih(x)
        out = self.relu(out)
        out = self.fcl_ho(out)

        return out


input_dim = 28*28
hidden_dim = 256
output_dim = 10

model = MLP(input_dim, hidden_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
iter = 0
losses = []


for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(images.shape[0], 1, 28, 28).float()
        
        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)
        print(loss)

        loss.backward()

        optimizer.step()

        # print('loss: ', loss.data.item())
        losses.append(loss.data.item())
        if i % 50 == 0:
            print(loss.data.item())


total = 0
correct = 0
for images, labels in test_loader:
    # Forward pass only to get logits/output
    images = images.view(images.shape[0], 1, 28, 28).float()
    outputs = model(images)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum()

accuracy = 100 * correct / total

print('Accuracy: ',accuracy.item(), '%')