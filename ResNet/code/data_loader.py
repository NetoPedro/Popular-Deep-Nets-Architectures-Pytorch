import torchvision
import torch

class To3Channels(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample.shape[0] < 3:
            sample = torch.squeeze(sample)
            sample = torch.stack([sample, sample,sample], 0)

        return sample

transformer_cifar_10 =  torchvision.transforms.Compose(
    [torchvision.transforms.Resize(224),
     torchvision.transforms.ToTensor(),
     To3Channels(),
     #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transformer_cifar_100 = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(224),
     torchvision.transforms.ToTensor(),
     To3Channels(),
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
     #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformer_fashion = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(224),
     torchvision.transforms.ToTensor(),
     To3Channels(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

CIFAR10_train = torchvision.datasets.CIFAR10("../datasets/CIFAR10/", train=True, transform=transformer_cifar_10, target_transform=None, download=True)
CIFAR100_train = torchvision.datasets.CIFAR100("../datasets/CIFAR100/", train=True, transform=transformer_cifar_100, target_transform=None, download=True)
FashionMNIST_train = torchvision.datasets.FashionMNIST("../datasets/FashionMNIST/", train=True, transform=transformer_fashion, target_transform=None, download=True)




CIFAR10_test = torchvision.datasets.CIFAR10("../datasets/CIFAR10/", train=False, transform=transformer_cifar_10, target_transform=None, download=True)
CIFAR100_test = torchvision.datasets.CIFAR100("../datasets/CIFAR100/", train=False, transform=transformer_cifar_100, target_transform=None, download=True)
FashionMNIST_test = torchvision.datasets.FashionMNIST("../datasets/FashionMNIST/", train=False, transform=transformer_fashion, target_transform=None, download=True)

def get_loaders(dataset = "CIFAR10"):
    train_loader = None
    test_loader = None
    labels_num = None

    if dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=64,
                                          shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=64,
                                          shuffle=True, num_workers=2)
        labels_num = 10#len(set(CIFAR10_train.train_labels))
    elif dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(CIFAR100_train, batch_size=64,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=64,
                                                  shuffle=True, num_workers=2)
        labels_num = 100

    elif dataset == "FASHION_MNIST":
        train_loader = torch.utils.data.DataLoader(FashionMNIST_train, batch_size=64,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(FashionMNIST_test, batch_size=64,
                                                  shuffle=True, num_workers=2)
        labels_num = len(set(FashionMNIST_train.train_labels))


    return train_loader,test_loader,labels_num


