import data_loader
import trainer
import network
import torch
import matplotlib.pyplot as mlp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def run(dataset = "CIFAR10",epochs = 30):
    trainloader, testloader, num_classes = data_loader.get_loaders(dataset)

    net = network.VGG16(num_classes)
    net.to(device)

    accuracies, losses = trainer.train(net, trainloader, testloader,epochs = epochs)

    x = np.linspace(0, 1, len(losses))

    mlp.plot(losses, x)
    mlp.show()

    x = np.linspace(0, 1, len(accuracies))

    mlp.plot(accuracies, x )
    mlp.show()

run(epochs=15)
run("CIFAR100",30)
run("FASHION_MNIST",5)


