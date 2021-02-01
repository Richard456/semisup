import random
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import sys
#sys.path.append("../")
from data_utils import ADVMNISTLoader
from losses import trades_loss
from attack_pgd import pgd
from smoothing import quick_smoothing
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# from test import test

source_dataset_name = 'MNIST'
target_dataset_name = 'pseudo_labeled_adv_mnistm'
data_root = "../dataset"
source_image_root = os.path.join(data_root, source_dataset_name)
target_image_root = os.path.join(data_root, target_dataset_name)
model_root = os.path.join('saved_models')
cudnn.benchmark = True

lr = 3e-4
batch_size = 128
image_size = 28
n_epoch = 100

beta = 1
weight = 1

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)
log_dir = 'runs/robust_self_training'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok = True)
logger = SummaryWriter(log_dir)

# load data
def data_load():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset_source = datasets.MNIST(
        root=data_root,
        train=True,
        transform=img_transform,
        download=True
    )

    train_dataloader_source = torch.utils.data.DataLoader(
        dataset=train_dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)


    train_dataset_target = ADVMNISTLoader(
        data_path=os.path.join(target_image_root, 'train.npy'),
        transform=img_transform
    )


    train_dataloader_target = torch.utils.data.DataLoader(
        dataset=train_dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    test_dataset_source = datasets.MNIST(
        root=data_root,
        train=False,
        transform=img_transform,
        download=True
    )

    test_dataloader_source = torch.utils.data.DataLoader(
        dataset=test_dataset_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

    test_dataset_target = ADVMNISTLoader(
        data_path=os.path.join(target_image_root, 'test.npy'),
        transform=img_transform
    )

    test_dataloader_target = torch.utils.data.DataLoader(
        dataset=test_dataset_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

    return train_dataloader_source, train_dataloader_target, test_dataloader_source, test_dataloader_target


# load model
model = torch.load("saved_models/mnist_model.pth")
model = model.cuda()

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# CEL = torch.nn.CrossEntropyLoss()
# KLL = torch.nn.KLDivLoss()
# CEL = CEL.cuda()
# KLL = KLL.cuda()


def train_one_epoch(model, dataloader_source, dataloader_target, epoch, args):
    model.train()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i+1 < len_dataloader:

        # training model using source data
        s_img, s_label = data_source_iter.next()
        s_img = s_img.expand(s_img.data.shape[0], 3, 28, 28)
        s_img = s_img.cuda()
        s_label = s_label.cuda()

        # training model using target data
        t_img, t_label = data_target_iter.next()
        t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)
        t_img = t_img.cuda()
        t_label = t_label.cuda()

        # s_class_output= model(input_data=s_img)
        # t_class_output = model(input_data=t_img)
        # 
        # loss_s = CEL(s_class_output, s_label)
        # robust_loss_t = CEL(t_class_output, t_label) + beta * KLL(F.log_softmax(t_class_output, dim=1), F.softmax(s_class_output, dim=1))
        # 
        # loss = loss_s + weight * robust_loss_t

        cat_img = torch.cat((s_img, t_img), 0)
        cat_target = torch.cat((s_label, t_label), 0)

        (loss, natural_loss, robust_loss,
         entropy_loss_unlabeled) = trades_loss(
            model=model,
            x_natural=cat_img,
            y=cat_target,
            optimizer=optimizer,
            step_size=args.pgd_step_size,
            epsilon=args.epsilon,
            perturb_steps=args.pgd_num_steps,
            beta=args.beta,
            distance=args.distance,
            adversarial=args.distance == 'l_inf',
            entropy_weight=args.entropy_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log to tensorboard
        global_step = epoch * (len_dataloader-1) + i
        logger.add_scalar("loss", loss, global_step)

        i += 1

        if i % 100 == 0:
            print('epoch: %d, [iter: %d / all %d], loss: %f' \
                  % (epoch, i, len_dataloader, loss.cpu().data.numpy()))


def test(model, dataloader, dataset_name, epoch, args):
    model.eval()
    model.cuda()

    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    with torch.no_grad():
        for batch_idx, (t_img, t_label) in enumerate(dataloader):
            t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)
            t_img = t_img.cuda()
            t_label = t_label.cuda()

            class_output= model(input_data=t_img)
            loss += F.cross_entropy(class_output, t_label, reduction='sum').item()
            pred = class_output.data.max(1, keepdim=True)[1]
            correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            if batch_idx < args.eval_attack_batches:
                if args.distance == 'l_2':
                    # run coarse certification
                    incorrect_clean, incorrect_rob = quick_smoothing(
                        model, t_img, t_label,
                        sigma=args.epsilon,
                        eps=args.epsilon,
                        num_smooth=100, batch_size=1000)
                    pass
                elif args.distance == 'l_inf':
                    # run medium-strength gradient attack
                    is_correct_clean, is_correct_rob = pgd(
                        model, t_img, t_label,
                        epsilon=args.epsilon,
                        num_steps=2 * args.pgd_num_steps,
                        step_size=args.pgd_step_size,
                        random_start=False)
                    incorrect_clean = (1 - is_correct_clean).sum()
                    incorrect_rob = (1 - np.prod(is_correct_rob, axis=1)).sum()
                else:
                    raise ValueError('No support for distance %s',
                                     args.distance)
                adv_correct_clean += (len(t_img) - int(incorrect_clean))
                adv_correct += (len(t_img) - int(incorrect_rob))
                adv_total += len(t_img)

            total += len(t_img)

    loss /= total
    accuracy = correct.data.numpy() * 1.0 / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    # eval_data = dict(loss=loss, accuracy=accuracy,
    #                  robust_accuracy=robust_accuracy,
    #                  robust_clean_accuracy=robust_clean_accuracy)
    # eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}

    logger.add_scalar("Clean loss", loss, epoch)
    logger.add_scalar("Clean accuracy", accuracy, epoch)
    logger.add_scalar("{} Clean accuracy".format('Smoothing' if args.distance == 'l_2' else 'PGD'), robust_clean_accuracy, epoch)
    logger.add_scalar("Robust accuracy", robust_accuracy, epoch)

    print(
        'Epoch {}:  Clean loss: {:.4f}, Clean accuracy: {}/{} ({:.2f}%), {} clean accuracy: {}/{} ({:.2f}%), Robust accuracy {}/{} ({:.2f}%)'.format(
            epoch, loss,
            correct, total, 100.0 * accuracy,
            'Smoothing' if args.distance == 'l_2' else 'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # parser.add_argument('--eps', default=0.3, type=float, help='eps')
    parser.add_argument('--distance', '-d', default='l_2', type=str,
                        help='Metric for attack model: l_inf uses adversarial '
                             'training and l_2 uses stability training and '
                             'randomized smoothing certification',
                        choices=['l_inf', 'l_2'])
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='Adversarial perturbation size (takes the role of'
                             ' sigma for stability training)')
    parser.add_argument('--pgd_num_steps', default=10, type=int,
                        help='number of pgd steps in adversarial training')
    parser.add_argument('--pgd_step_size', default=0.007,
                        help='pgd steps size in adversarial training', type=float)
    parser.add_argument('--beta', default=6.0, type=float,
                        help='stability regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--entropy_weight', type=float,
                        default=0.0, help='Weight on entropy loss')

    # Eval config
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Eval frequency (in epochs)')
    parser.add_argument('--eval_attack_batches', default=1, type=int,
                        help='Number of eval batches to attack with PGD or certify '
                             'with randomized smoothing')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)

    train_dataloader_source, train_dataloader_target, \
    test_dataloader_source, test_dataloader_target = data_load()

    # training
    for epoch in range(n_epoch):
        train_one_epoch(model, train_dataloader_source, train_dataloader_target, epoch, args)
        # test(model, test_dataloader_source, source_dataset_name, epoch, args)
        test(model, test_dataloader_target, target_dataset_name, epoch, args)

    torch.save(model, '{0}/robust_self_training.pth'.format(model_root))
