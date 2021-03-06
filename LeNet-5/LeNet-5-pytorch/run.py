from lenet import LeNet5, TestLeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom

viz = visdom.Visdom()

data_train = MNIST(r'F:\1.mnist\input',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))
data_test = MNIST(r'F:\1.mnist\input',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((28, 28)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=0)



# 生成一个模型
#net = LeNet5()
net = TestLeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images), Variable(labels)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.FloatTensor(loss_list), torch.FloatTensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.item(), float(total_correct) / len(data_test)))



# 训练和测试主体
def train_and_test(epoch):
    train(epoch)
    test()



# LeNet-5主函数
# 16表示的epoch, 这个可以提前设置
#
def main():
    for e in range(1, 16):
        train_and_test(e)


# 运行的主函数入口
if __name__ == '__main__':
    main()
