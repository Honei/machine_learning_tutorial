import torch.nn as nn
from torch.autograd import Variable
import torch


basic_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=1)
toy_input = Variable(torch.randn(3, 8, 4))
h_0 = Variable(torch.randn(1, 8, 3))

toy_output, h_n = basic_rnn(toy_input, h_0)
print('toy_output.size(): ', toy_output.size())
print('hn_size(): ', h_n.size())
print('hn', h_n[0][0])
print('output[2]', toy_output[2][0])