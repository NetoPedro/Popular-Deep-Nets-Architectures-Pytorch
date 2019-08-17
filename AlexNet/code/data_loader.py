import torchvision
import torch
transformer =  torchvision.transforms.Compose(
    [torchvision.transforms.Resize(224),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

CIFAR10_train = torchvision.datasets.CIFAR10("../datasets/CIFAR10/", train=True, transform=transformer, target_transform=None, download=True)
CIFAR100_train = torchvision.datasets.CIFAR100("../datasets/CIFAR100/", train=True, transform=transformer, target_transform=None, download=True)
FashionMNIST_train = torchvision.datasets.FashionMNIST("../datasets/FashionMNIST/", train=True, transform=transformer, target_transform=None, download=True)




CIFAR10_test = torchvision.datasets.CIFAR10("../datasets/CIFAR10/", train=False, transform=transformer, target_transform=None, download=True)
CIFAR100_test = torchvision.datasets.CIFAR100("../datasets/CIFAR100/", train=False, transform=transformer, target_transform=None, download=True)
FashionMNIST_test = torchvision.datasets.FashionMNIST("../datasets/FashionMNIST/", train=False, transform=transformer, target_transform=None, download=True)

def get_loaders(dataset = "CIFAR10"):
    train_loader = None
    test_loader = None
    labels_num = None

    if dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=4,
                                          shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=4,
                                          shuffle=True, num_workers=2)
        labels_num = len(set(CIFAR10_train.train_labels))
    elif dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(CIFAR100_train, batch_size=4,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=4,
                                                  shuffle=True, num_workers=2)
        labels_num = len(set(CIFAR100_train.train_labels))

    elif dataset == "FASHION_MNIST":
        train_loader = torch.utils.data.DataLoader(FashionMNIST_train, batch_size=4,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(FashionMNIST_test, batch_size=4,
                                                  shuffle=True, num_workers=2)
        labels_num = len(set(FashionMNIST_train.train_labels))


    return train_loader,test_loader,labels_num