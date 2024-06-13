from cifar10_models import *
# from approximate_gradients import *
import network

def get_classifier_cifar(classifier, pretrained=True, num_classes=10):
    if classifier.lower()=='lenet5':
        return network.lenet.LeNet5()
    elif classifier.lower()=='lenet5half':
        print('using lenet5half')
        return network.lenet.LeNet5Half()
    elif classifier.lower()=='lenet5fifth':
        print('using lenet5fifth')
        return network.lenet.LeNet5Fifth()
    elif classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg16_bn':
        print('using VGG16_BN network as teacher network')
        return vgg16_bn(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(pretrained=pretrained, num_classes=num_classes)
    if classifier == 'vgg11':
        return vgg11(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg13':
        return vgg13(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg16':
        print('using VGG16 as teacher network')
        return vgg16(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'vgg19':
        return vgg19(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet18':
    #     return resnet18(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet34':
    #     return resnet34(pretrained=pretrained, num_classes=num_classes)
    # elif classifier == 'resnet50':
    #     return resnet50(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet121':
        print('Using densenet121 as the student network')
        return densenet121(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet161':
        return densenet161(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'densenet169':
        return densenet169(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'mobilenet_v2':
        print('using mobilenet_v2')
        return mobilenet_v2(pretrained=pretrained, num_classes=num_classes)
    elif classifier == 'googlenet':
        print('using googlenet')
        # print('using classess', num_classes)
        return GoogLeNet(num_classes=num_classes)
    elif classifier == 'inception_v3':
        print('using inception_v3')
        return Inception3(num_classes=num_classes)

    elif classifier == "wrn-28-10":
        net =  wrn(
                    num_classes=num_classes,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3
                )
        if pretrained:
            state_dict = torch.load("cifar100_models/state_dicts/model_best.pt", map_location=device)["state_dict"]
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        return net
    elif 'wrn' in classifier and 'kt' not in classifier:
        depth = int(classifier.split("-")[1])
        width = int(classifier.split("-")[2])

        net =  wrn(
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=width
                )
        if pretrained:
            raise ValueError("Cannot be pretrained")
        return net
    elif classifier == "kt-wrn-40-2":
        net = WideResNetKT(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        if pretrained:
            state_dict = torch.load("cifar10_models/state_dicts/kt_wrn.pt", map_location=device)["state_dict"]
            net.load_state_dict(state_dict)
        return net

    elif classifier == "kt-wrn-28-2-new":
        print('loading classifier of', classifier)
        net = WideResNetKTnew(depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        # if pretrained:
        #     state_dict = torch.load("cifar10_models/state_dicts/kt_wrn.pt", map_location=device)["state_dict"]
        #     net.load_state_dict(state_dict)
        return net
    elif classifier == "resnet50_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet50_8x(num_classes=num_classes)
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        print('loading resnet18_8X')
        return network.resnet_8x.ResNet18_8x(num_classes=num_classes)

    else:
        raise NameError('Please enter a valid classifier')


# def measure_true_grad_norm(args, x):
#     # Compute true gradient of loss wrt x
#     true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
#     true_grad = true_grad.view(-1, 3072)

#     # Compute norm of gradients
#     norm_grad = true_grad.norm(2, dim=1).mean().cpu()

#     return norm_grad

classifiers = [
    'lenet5',
    'lenet5half',
    'lenet5fifth',
    "resnet50_8x",
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
    "kt-wrn-28-2",
    "kt-wrn-28-2-new",
]