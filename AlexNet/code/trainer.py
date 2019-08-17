import torch
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(i)
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                losses.append(running_loss)
                running_loss = 0.0
