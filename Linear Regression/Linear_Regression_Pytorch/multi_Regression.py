# 多项式回归
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 真实的系数和偏置
w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def make_features(x):
    x = x.unsqueeze(1)
    x = torch.cat([x ** i for i in range(1, 4)], 1)
    return x

def f(x):
    x = x.mm(w_target) + b_target[0]
    return x

# 1. 获取数据,通过这个函数可以获取训练数据
def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)

    return Variable(x), Variable(y)


# 2. 设计模型
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly_regression = nn.Linear(3, 1)

    def forward(self, x):
        x = self.poly_regression(x)
        return x

# 3. 创建模型实例
model = poly_model()

# 4. 设计判卷准则
criterion = nn.MSELoss()

# 5. 使用优化算法
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    x_train, y_train = get_batch()

    # 6. 获取模型的输出值
    out = model(x_train)

    # 7. 得到损失函数
    loss = criterion(y_train, out)

    # 8. 清空参数的所有参数
    optimizer.zero_grad()

    # 9. 计算损失的梯度
    loss.backward()

    # 10. 更新参数
    optimizer.step()

    epoch += 1
    if epoch % 20 == 0:
        print('| Epoch {}, loss: {:.6f}'
              .format(epoch, loss.item()))

    if loss <= 1e-3:
        break


# 11. 准备测试数据
x_train, y_train = get_batch()

# 12. 切换到测试模式
model.eval()

# 13. 获取测试结果
predict = model(x_train)
predict = predict

# 14. 绘制所有数据
x_train = x_train[:, 0].view(32,1)
print('x_train: ', x_train.size())
print('y_train: ', y_train.size())
print('predict: ', predict.size())
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'ro', label='Original data')
plt.plot(x_train.data.numpy(), predict.data.numpy(), 'b*', label='Fitting Line')
plt.show()

