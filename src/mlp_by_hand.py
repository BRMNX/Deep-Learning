import torch
from matplotlib import pyplot as plt
import matplotlib
from d2l import torch as d2l
from torch import nn
matplotlib.use('TkAgg')
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # @表示矩阵乘法
    return (H @ W2 + b2)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_hiddens = 256
num_outputs = 10
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs)) 
params = [W1, b1, W2, b2]
loss = nn.CrossEntropyLoss(reduction='none')
num_eopchs = 10
lr = 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_eopchs, updater)
plt.show()
d2l.predict_ch3(net, test_iter)
plt.show()