import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Import the network/backbones to be used
from networks.backbones.SimpleNet import SimpleNet
# from networks.backbones.QuanvNet import SimpleNet

parser = argparse.ArgumentParser()
parser.add_argument('--floq_key', default=None, type=str, help='Your Floq Api Key')
parser = parser.parse_args()

floq_key = parser.floq_key


def test():
    # Testing
    t0 = time.time()
    correct = 0
    total = 0
    avg_loss = 0 
    num_samples = 1000
    for i, (images, labels) in enumerate(train_loader):
        if i == num_samples:
            break
    
        images = images.to(device)
        labels = labels.to(device)

        if labels.item()==6:
            labels = torch.tensor([1])
        if labels.item()==9:
            labels = torch.tensor([2])

        out = net(images)
        _, predicted_labels = torch.max(out, 1)
        loss = loss_fun(out, labels).to(device)
        avg_loss += loss.item()
        correct += (predicted_labels == labels).sum()
        total += labels.size(0)

    avg_loss = avg_loss / num_samples 

    with open('tools/eval_stats/log_validation.csv', 'a') as f:
        f.write('%.4f, %.4f\n' %((100.0 * correct) / (total + 1), avg_loss))

    print('Percent correct: %.3f' % ((100.0 * correct) / (total + 1)))

    print('Loss: %.4f' %avg_loss)

    print("---Testing took %s seconds ---" % (time.time() - t0))

# Define the "device". If GPU is available, device is set to use it, otherwise CPU will be used. 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# To randomly transform the image.
rand_transform = transforms.Compose([transforms.RandomChoice([
    transforms.Pad(3),
    transforms.RandomCrop(26),
    transforms.Pad(1),
    transforms.RandomCrop(27),
]), transforms.ToTensor()])

# To download and setup the train/test dataset
train_data = datasets.MNIST(root='./data', train=True,
                            transform=rand_transform, download=True)

test_data = datasets.MNIST(root='./data', train=False,
                           transform=rand_transform, download=True)

idx = (train_data.targets == 0) | (train_data.targets == 6) | (train_data.targets == 9)
train_data.targets = train_data.targets[idx]
train_data.data = train_data.data[idx]

idx = (test_data.targets == 0) | (test_data.targets == 6) | (test_data.targets == 9)
test_data.targets = test_data.targets[idx]
test_data.data = test_data.data[idx]

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Loading the model
net = SimpleNet().to(device)
#print(net)

# Preparation for training
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1.e-3)

# Training
num_epochs = 5
num_iters_per_epoch = 1000  # use only 5K iterations
N_TEST = 5
print("Beginning Training")
start_time = time.time()

if not os.path.exists('tools/eval_stats'):
    os.makedirs('tools/eval_stats')
if not os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')

try:
    os.remove('tools/eval_stats/log_validation.csv')
except OSError:
    pass

test()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if i == num_iters_per_epoch:
            break
        images = images.to(device)
        labels = labels.to(device)
        
        if labels.item()==6:
            labels = torch.tensor([1])
        if labels.item()==9:
            labels = torch.tensor([2])

        optimizer.zero_grad()
        output = net(images)

        loss = loss_fun(output, labels).to(device)
        loss.backward()
        optimizer.step()

        if (i+1)%(num_iters_per_epoch//N_TEST)==0:
            test()

        if (i + 1) % (num_iters_per_epoch // 10) == 0:
            print('Epoch [%d/%d], Step [%d/%d]'
                  % (epoch + 1, num_epochs, i + 1, num_iters_per_epoch))

    print("Saving Checkpoint for Epoch", epoch + 1)
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'pretrained_models/QuanvNet-epoch_{}.pt'.format(epoch + 1))
    print("---Epoch Time: %s seconds ---" % (time.time() - start_time))

train_time = time.time()
print("---Training took %s seconds ---" % (train_time - start_time))


