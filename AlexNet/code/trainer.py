import torch
from torch import optim


def train(net,trainloader,testloader,optim_name = "adam"):
    optimizer = optim.Adam(net.parameters(), 0.001)
    if optim_name == "sgd":
        optimizer = optim.SGD(net.parameters(),0.001,0.9)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = 2
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                losses.append(running_loss)
                running_loss = 0.0
