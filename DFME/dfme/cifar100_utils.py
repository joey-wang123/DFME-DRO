from torchvision import datasets, transforms
import torch
import os

import torch
import torch.nn.functional as F
import torch.nn as nn

def myprint(a):
    """Log the print statements"""
    global file
    # print(a);
    log.info(a);
    file.write(a);
    file.write("\n");
    file.flush()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize_coefs=None, normalize=False):
        super(ResNet, self).__init__()

        if normalize_coefs is not None:
            self.mean, self.std = normalize_coefs

        self.normalize = normalize

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):

        if self.normalize:
            # Normalize according to the training data normalization statistics
            x -= self.mean
            x /= self.std

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out, feature


def ResNet18_8x(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34_8x(num_classes=10, normalize_coefs=None, normalize=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, normalize_coefs=normalize_coefs, normalize=normalize)


def ResNet50_8x(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101_8x(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152_8x(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def get_classifier_new(classifier, pretrained=True, num_classes=10):
    if classifier == "wrn-28-10":
        pass
    elif 'wrn' in classifier and 'kt' not in classifier:
        pass
    elif classifier == "kt-wrn-40-2":
        pass
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return ResNet34_8x(num_classes=num_classes)
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        return ResNet18_8x(num_classes=num_classes)

    else:
        raise NameError('Please enter a valid classifier')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, activation=None, final_bn=True):
        super(GeneratorA, self).__init__()

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False)
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False)
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x:
            return img
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            return self.activation(img)


from approximate_gradients import *
def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad

classifiers = [
    "resnet34_8x", # Default DFAD
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "mobilenet_v2",
    "googlenet",
    "inception_v3",
    "wrn-28-10",
    "resnet18_8x",
    "kt-wrn-40-2",
]



def get_dataloader(args):
    if args.dataset.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset.lower() == 'svhn':
        print("Loading SVHN data")
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.43768206, 0.44376972, 0.47280434),
                                                   (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.43768206, 0.44376972, 0.47280434),
                                                   (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset.lower() == 'cifar10':
        print("Loading cifar10 data")
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif args.dataset.lower() == 'cifar100':
        print("Loading cifar100 data")
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=True, download=False,
                              transform=transforms.Compose([
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader