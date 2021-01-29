import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_utils import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import MNISTM_Model
import numpy as np

dataset_name = 'MNIST'
image_root = os.path.join('../dataset', dataset_name)
model_root = os.path.join('saved_models')
cudnn.benchmark = True
lr = 3e-4
batch_size = 128
image_size = 28
n_epoch = 50

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root='dataset',
    train=True,
    transform=img_transform,
    download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

test_dataset = datasets.MNIST(
    root='dataset',
    train=False,
    transform=img_transform,
    download=True
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)

# load model
model = MNISTM_Model()  #load mnistm model for three channels

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_func = torch.nn.CrossEntropyLoss()

model = model.cuda()
loss_func = loss_func.cuda()

def train_one_epoch(model, dataloader, epoch):
    model.train()

    for i, (img, label) in enumerate(dataloader):
        # training model using source data
        img = img.expand(img.data.shape[0], 3, 28, 28)
        img = img.cuda()
        label = label.cuda()
        output = model(input_data=img)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i+1, loss))

def test(model, dataloader, epoch):

    """ training """
    model = model.eval()
    model = model.cuda()

    # i = 0
    n_total = 0
    n_correct = 0

    for img, label in dataloader:
        img = img.expand(img.data.shape[0], 3, 28, 28)
        batch_size = img.shape[0]
        img = img.cuda()
        label = label.cuda()

        output = model(input_data=img)
        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('Epoch: {}, Test Acc: {}'.format(epoch, accu))

# training
for epoch in range(n_epoch):

    train_one_epoch(model, train_dataloader, epoch)
    test(model, test_dataloader, epoch)

torch.save(model, '{0}/mnist_model.pth'.format(model_root))
