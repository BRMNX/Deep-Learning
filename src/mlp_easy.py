import torch
from matplotlib import pyplot as plt
import matplotlib
from d2l import torch as d2l
from torch import nn
matplotlib.use('TkAgg')

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256)
batch_size, num_epochs, lr = 256, 10, 0.1
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
plt.show()