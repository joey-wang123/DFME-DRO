#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from lenet import LeNet5
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from perturbnet import *
from my_utils import *
import argparse
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import time

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',help='input batch size for training (default: 256)')
parser.add_argument('--T_step', type=int, default=1, metavar='N',help='number of evolution steps')
parser.add_argument('--noisenet-max-eps', default=0.6, type=float)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--stepsize', type=float, default=0.0001, help="memory evolution learning rate")
parser.add_argument('-m','--method', type=str, default='SVGD', choices=['SVGD'])
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers, help='Target model name (default: resnet34_8x)')
args = parser.parse_args()

# os.makedirs(args.output_dir, exist_ok=True)  
num_classes = 10 if args.dataset in ['cifar10', 'svhn'] else 100
args.num_classes = num_classes

acc = 0
acc_best = 0

if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                       transform=transform_train,
                       download=True)
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test,
                      download=True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

    net = get_classifier(args, args.model, pretrained=False, num_classes=args.num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                       transform=transform_train,
                       download=True)
    data_test = CIFAR100(args.data,
                      train=False,
                      transform=transform_test,
                      download=True)
                      
    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)
    # net = resnet.ResNet34(num_classes=100).cuda()


    net = get_classifier(args, args.model, pretrained=False, num_classes=args.num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # if epoch < 80:
    #     lr = 0.1
    # elif epoch < 120:
    #     lr = 0.01
    # else:
    #     lr = 0.001

    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    elif epoch < 160:
        lr = 0.001
    else:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def SVGD_kernal(flat_x, h=-1):

    x_numpy = flat_x.cpu().data.numpy()
    init_dist = pdist(x_numpy)
    pairwise_dists = squareform(init_dist)

    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = 0.05 * h ** 2 / np.log(flat_x.shape[0] + 1)

    if x_numpy.shape[0] > 1:
        kernal_xj_xi = torch.exp(- torch.tensor(pairwise_dists) ** 2 / h)
    else:
        kernal_xj_xi = torch.tensor([1])

    return kernal_xj_xi, h

def SVGD_step(stepsize, z_gen, target_grad):
    """z_gen is the memory data """
    """ target_grad is the gradient of per datapoint in memory buffer"""
    device = target_grad.device
    kernal_xj_xi, h = SVGD_kernal(z_gen, h=-1)
    kernal_xj_xi, h = kernal_xj_xi.float(), h.astype(float)
    kernal_xj_xi = kernal_xj_xi.to(device)

    d_kernal_xi = torch.zeros(z_gen.size()).to(device)
    x = z_gen

    for i_index in range(x.size()[0]):
        d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

    current_grad = (torch.matmul(kernal_xj_xi, target_grad) + d_kernal_xi) / x.size(0)
    WGF_sample = z_gen - stepsize * current_grad
    return WGF_sample

        
def train(epoch, scale, perturbnet):

    L1loss = nn.L1Loss()

    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        if args.batch_size == images.size(0):
            perturbnet.reload_parameters()
            perturbnet.set_epsilon(random.uniform(args.noisenet_max_eps / 2.0, args.noisenet_max_eps))

            if args.dataset == 'MNIST':
                images = images.repeat(1, 3, 1, 1)
            image_view = images.view(1, -1, 32, 32)

            perturbation = perturbnet(image_view)
            perturbation = perturbation.view(args.batch_size, 3, 32, 32)


            z_hat = deepcopy(images)
            z_hat = z_hat.cuda()
            z_hat = z_hat.clone().detach().requires_grad_(True)

            for n in range(args.T_step):
                delta = z_hat - images
                rho = torch.mean((torch.norm(delta.view(len(images), -1), 2, 1) ** 2))
                loss_zt = F.cross_entropy(net(z_hat), labels)
                loss_phi = - (loss_zt - args.gamma * rho)
                loss_phi.backward()
                target_grad = z_hat.grad



                if args.method == 'SVGD':
                    input_shape = z_hat.size()
                    flat_z = z_hat.view(input_shape[0], -1)
                    target_grad = target_grad.view(input_shape[0], -1)
                    flat_z = SVGD_step(args.stepsize, flat_z, target_grad)
                    z_hat = flat_z.view(list(input_shape))

                z_hat = z_hat.clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            output_train = net(images + scale*perturbation)


            new_zhat = z_hat.view(1, -1, 32, 32)
            zhat_noise = perturbnet(new_zhat)
            zhat_noise = zhat_noise.view(args.batch_size, 3, 32, 32)
            output_val = net(z_hat + scale*zhat_noise)
            ori_val = net(z_hat)
            L1norm = L1loss(output_val, ori_val)
        
            output = net(images)
            loss = criterion(output, labels)

            loss_val = criterion(output_val, labels)
            loss += loss_val 

            loss += 0.001*L1norm

            loss_list.append(loss.data.item())
            batch_list.append(i+1)
    
            if i == 1:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
    
            loss.backward()
            optimizer.step()
 
 
def test(scale, perturbnet):
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0

    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            if args.dataset == 'MNIST':
                    images = images.repeat(1, 3, 1, 1)
            if args.batch_size == images.size(0):
                    perturbnet.reload_parameters()
                    perturbnet.set_epsilon(random.uniform(args.noisenet_max_eps / 2.0, args.noisenet_max_eps))

                    image_view = images.view(1, -1, 32, 32)
                    perturbation = perturbnet(image_view)
                    perturbation = perturbation.view(args.batch_size, 3, 32, 32)
                    output = net(images + scale*perturbation)
                    total += images.size(0)

                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)

    if acc_best < acc:
        acc_best = acc

    print(f'Test Avg. Loss: {avg_loss.data.item()}, Accuracy: {acc}')
    



def test_utility(scale, perturbnet):
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0

    diff_list = []
    total = 0
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            if args.dataset == 'MNIST':
                    images = images.repeat(1, 3, 1, 1)
            if args.batch_size == images.size(0):
                    perturbnet.reload_parameters()
                    perturbnet.set_epsilon(random.uniform(args.noisenet_max_eps / 2.0, args.noisenet_max_eps))

                    image_view = images.view(1, -1, 32, 32)
                    perturbation = perturbnet(image_view)
                    perturbation = perturbation.view(args.batch_size, 3, 32, 32)

                    output = net(images + scale*perturbation)
                    clean_output = net(images)
                    noise_prob = softmax(output)
                    clean_prob = softmax(clean_output)
                    diff = noise_prob - clean_prob
                    norm_diff = torch.norm(diff, p=1, dim=1)

                    total += images.size(0)
                    diff_list.append(norm_diff)

                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)

    print(f'Test Avg. Loss: {avg_loss.data.item()}, Accuracy: {acc}')

 
 
def train_and_test(epoch, scale, perturbnet):
    train(epoch, scale, perturbnet = perturbnet)
    test(scale, perturbnet = perturbnet)

 
 
def main():

    scale = 0.5
    perturbnet_batch_size = int(args.batch_size)
    PATH = args.output_dir + f'{args.model}/newteacher_{args.dataset}_{args.method}_step{args.T_step}_scale_{scale}/'
    perturbnet = Res2Net(epsilon=0.50, hidden_planes=2, batch_size=perturbnet_batch_size).train().cuda()

    if not os.path.exists(PATH):
        os.makedirs(PATH)


    if args.mode == 'train':
        if args.dataset == 'MNIST':
            epoch = 50
        else:
            epoch = 200

        if not os.path.exists(PATH):
            os.makedirs(PATH)

        for e in range(1, epoch):
            train_and_test(e, scale, perturbnet = perturbnet)
            if e>150 and e%5==0:
                torch.save(net.state_dict(), PATH + f'{e}.pth')

    elif args.mode == 'test':
        if args.dataset == 'MNIST':
            e = 9
        else:
            e = 195
        load_PATH =  PATH + f'{e}.pth'
        net.load_state_dict(torch.load(load_PATH))
        test_utility(scale, perturbnet)

 
if __name__ == '__main__':
    main()