# 一元线性回归

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.86], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

assert x_train.size == y_train.size

## 1.准备数据
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

## 2.设计模型
class SingleLinearRegression(nn.Module):
    def __init__(self):
        super(SingleLinearRegression, self).__init__()
        self.regression = nn.Linear(1, 1)

    def forward(self, x):
        out = self.regression(x)

        return out

## 3. 创建模型实例
model = SingleLinearRegression()

## 4. 设计判决准则
criterion = nn.MSELoss()

## 5. 使用优化方法
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 2000
for i in range(epoch):
    x_train = Variable(x_train)
    y_train = Variable(y_train)

    ## 6. 获取模型的输出值
    out = model(x_train)

    ## 7. 得到损失函数值
    loss = criterion(y_train, out)

    ## 8. 清空参数的所有梯度
    optimizer.zero_grad()

    ## 9. 计算梯度值
    loss.backward()

    ## 10. 跟新参数
    optimizer.step()

    if i % 100 == 0:
        print('| Epoch[ {} / {} ], loss: {:.6f}'
              .format(i + 1, epoch, loss.item()))


## 11. 准备测试数据
x_train = Variable(x_train)

## 12. 切换到测试模型
model.eval()

## 13. 获取测试结果
predict = model(x_train)
predict = predict.data.numpy()

## 14.绘制所有数据
plt.plot(x_train.data.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.data.numpy(), predict, label='Fitting Line')
plt.show()