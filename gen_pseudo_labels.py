import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_utils import GetLoader
from data_utils import ADVMNISTLoader
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np
import functools
import operator


dataset_name = 'adv_mnistm'
eps=8/255
image_root = os.path.join('../dataset',dataset_name)
model_root = os.path.join('saved_models')
cudnn.benchmark = True
batch_size = 200
image_size = 28

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ADVMNISTLoader(
    data_path=os.path.join(image_root, 'train_eps{}.npy'.format(eps)),
    transform=img_transform
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8)

test_dataset = ADVMNISTLoader(
    data_path=os.path.join(image_root, 'test_eps{}.npy'.format(eps)),
    transform=img_transform
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8)

# load pretrained model
model = torch.load("saved_models/mnist_model.pth")
model = model.cuda()

train_imgs = []
train_pseudo_labels = []

test_imgs = []
test_labels = []


def gen_pseudo_labels(model, dataloader):
    model.eval()

    for i, (img, label) in enumerate(dataloader):

        img = img.expand(img.data.shape[0], 3, 28, 28)
        img = img.cuda()
        output = model(input_data=img)
        pred = output.data.max(1, keepdim=True)[1]
        pred = functools.reduce(operator.iconcat, pred, [])
        pred = torch.stack(pred).cpu().detach().numpy()

        train_imgs.extend(img.cpu().detach().numpy())
        train_pseudo_labels.extend(pred)

        print('processed {}'.format(i*batch_size))

def test(model, dataloader):

    model.eval()

    # i = 0
    n_total = 0
    n_correct = 0

    for img, label in dataloader:
        batch_size = img.shape[0]
        img = img.expand(img.data.shape[0], 3, 28, 28)
        img = img.cuda()
        label = label.cuda()

        test_imgs.extend(img.cpu().detach().numpy())
        test_labels.extend(label.cpu().detach().numpy())

        output = model(input_data=img)
        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('test Acc: {}'.format(accu))

# generating pseudo labels
test(model, test_dataloader)
gen_pseudo_labels(model, train_dataloader)

save_path = '../dataset/pseudo_labeled_adv_mnistm'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok = True)

np.save(os.path.join(save_path, 'train'), [train_imgs, train_pseudo_labels])
np.save(os.path.join(save_path, 'test'), [test_imgs, test_labels])


