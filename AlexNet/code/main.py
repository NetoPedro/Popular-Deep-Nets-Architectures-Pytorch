import data_loader
import trainer
import network
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


trainloader,testloader,num_classes = data_loader.get_loaders("FASHION_MNIST")

net = network.AlexNet(num_classes)
net.to(device)

trainer.train(net,trainloader,testloader)