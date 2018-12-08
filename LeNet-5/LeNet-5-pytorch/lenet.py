# LeNet-5 基于pytorch的实现
# 原作者连接： https://github.com/activatedgeek/LeNet-5
# 修改：熊哈哈


import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),    # 6 * 28 * 28
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), # 6 * 14 * 14
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),       # 16 * 10 * 10
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), # 16 * 5 * 5
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),     # 这还是一个卷积，卷积核是5*5
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),         #   全连接
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.Softmax())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output


class TestLeNet5(nn.Module):
    def __init__(self):
        super(TestLeNet5, self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ("C1", nn.Conv2d(1, 6, (5, 5))),       # 6 * 28 * 28
            ("S2", nn.MaxPool2d((2, 2), 2))        # 6 * 14 * 14
        ]))

        self.layer2 = nn.Sequential(OrderedDict([
            (("C3", nn.Conv2d(6, 16, (5, 5)))),     # 16 * 10 * 10
            (("S4", nn.MaxPool2d((2, 2), 2)))       # 16 * 5 * 5
        ]))

        self.C5 = nn.Conv2d(16, 120, (5, 5))        # 120 * 1 * 1

        self.F6 = nn.Linear(120, 84)

        self.F7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.C5(x)
        x = x.view(x.size()[0], -1)
        x = self.F6(x)
        x = self.F7(x)
        return x



'''
这个LeNet5的网络输入的图片尺寸是28*28
'''
class LeNet5_2(nn.Module):
    def __init__(self):
        super(LeNet5_2, self).__init__()
        
        # 使用nn.Sequential()保存第一层的网络
        self.layer1 = nn.Sequential()
        
        # 第一层卷积的卷积核是3*3， 使用padding，那么经过第一层卷积之后图片的维度是：
        # (28 - 3 + 2 * 1) / 1 + 1 = 28，即经过第一层卷积之后图片的尺寸还是 28 * 28
        self.layer1.add_module("C1", nn.Conv2d(1, 6, 3, padding=1))

        # 接下来使用一个池化层，池化选择2*2的最大池化，
        # 经过池化之后图片的尺寸是14*14
        self.layer1.add_module("S2", nn.MaxPool2d(2, 2))


        # 使用nn.Sequential()保存第二次层卷积
        self.layer2 = nn.Sequential()

        # 第二层卷积的卷积核是5*5，没有使用padding，那么经过第二层卷积之后图片的维度是
        # (14 - 5 ) / 1 + 1 = 10
        self.layer2.add_module('C3', nn.Conv2d(6, 16, (5, 5), padding=1))

        # 接下来使用一个池化层，池化选择2*2的最大池化
        # 经过池化之后图片的尺寸是5 * 5
        self.layer2.add_module('S4', nn.MaxPool2d(2, 2))


        # 下面使用一个全连接层,FC1没有使用卷积，直接使用了全连接
        self.layer3 = nn.Sequential()
        self.layer3.add_module('Fc1', nn.Linear(400, 120))
        self.layer3.add_module('Fc2', nn.Linear(120, 84))
        self.layer3.add_module('Fc3', nn.Linear(84, 10))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size()[0], -1)
        x = self.layer3(x)

        return x

if __name__ == "__main__":
    model = TestLeNet5()
    print(model)