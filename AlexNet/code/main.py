import data_loader
import trainer
import network
import torch
import matplotlib.pyplot as mlp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def run(dataset = "CIFAR10"):
    trainloader, testloader, num_classes = data_loader.get_loaders(dataset)

    net = network.AlexNet(num_classes)
    net.to(device)

    accuracies, losses = trainer.train(net, trainloader, testloader)

    x = np.linspace(0, 1, len(losses))

    mlp.plot(losses, x)
    mlp.show()

    x = np.linspace(0, 1, len(accuracies))

    mlp.plot(losses, accuracies)
    mlp.show()

run()
run("CIFAR100")
run("FASHION_MNIST")


