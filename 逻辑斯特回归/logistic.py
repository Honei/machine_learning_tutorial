import codecs
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1.准备数据
with codecs.open('./input/data.txt', 'r', 'utf-8')as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]



x0 = list(filter(lambda x: x[-1] == 0, data))
x1 = list(filter(lambda x: x[-1] == 1, data))

plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]

plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]


plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x_1')
plt.legend(loc='best')

x_data = [[i[0], i[1] ]for i in data]
y_data = [[i[2]] for i in data]
x_data = torch.Tensor(x_data)
y_data = torch.Tensor(y_data)

# 2.设计模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()      # logistic function

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)

        return x

# 3. 创建模型实例
model = LogisticRegression()


# 4. 设计判决准则
criterion = nn.BCELoss()

# 5. 使用优化方法
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(500000):
    # 6.获取一个batch的数据
    x_train = Variable(x_data)
    y_train = Variable(y_data)

    #7. 获取模型的输出值
    out = model(x_train)

    # 8. 计算损失值
    loss = criterion(out, y_train)

    # 9. 清空所有的梯度值
    optimizer.zero_grad()

    # 10. 计算梯度
    loss.backward()

    # 11. 更新参数
    optimizer.step()


    # 12. 获取决策数据
    mask = out.ge(0.5).float()
    correct = (mask == y_train).sum()
    acc = correct.item() / x_train.size(0)

    if (epoch + 1) % 1000 == 0:
        print('*'*10)
        print('| Epoch {}'.format(epoch+1))
        print('| loss: {:.6f}'.format(loss.item()))
        print('| acc {}'.format(acc))


# 13. 绘制决策平面
w0, w1 = model.lr.weight[0] # 取出所有的系数
w0 = w0.item()
w1 = w1.item()
b = model.lr.bias.item()

plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
plt.show()

