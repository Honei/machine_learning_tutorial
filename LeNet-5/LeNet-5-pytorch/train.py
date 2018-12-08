from lenet import TestLeNet5
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn


data_train = MNIST(r'F:\1.mnist\input', download=True,
                   transform=transforms.Compose([ transforms.Resize((32, 32)), transforms.ToTensor()])
                   )
data_test = MNIST(r'F:\1.mnist\input', train=False, download=True,
                  transform=transforms.Compose([ transforms.Resize((32, 32)),transforms.ToTensor()])
                  )
data_train_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=2, num_workers=8)

model = TestLeNet5()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

def main():
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images), Variable(labels)

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels).sum()

        loss.backward()

        optimizer.step()

        if i % 200 == 0:
            print("| loss = {}".format(loss.item()))
def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i , (images, labels) in enumerate(data_test_loader):
        images, labels = Variable(images), Variable(labels)
        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()


    avg_loss /= len(data_test)
    print("Test avg. Loss: {}, Accuracy: {}".format(avg_loss.item(), float(total_correct) / len(data_test)))


if __name__ == '__main__':
    main()
    test()