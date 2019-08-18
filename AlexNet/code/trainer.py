import torch
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net,trainloader,testloader,optim_name = "adam"):
    optimizer = optim.Adam(net.parameters(),lr= 0.001,weight_decay=0.0005)
    if optim_name == "sgd":
        optimizer = optim.SGD(net.parameters(),0.001,0.9)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = 30
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
            print(i)
            running_loss += loss.item()
            if i % 200 == 199:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                losses.append(running_loss)
                running_loss = 0.0

        accuracy = compute_accuracy(net,testloader)
        print('Accuracy of the network on the test images: %.3f' % accuracy)

def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total