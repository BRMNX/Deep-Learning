import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib
# 强制使用 TkAgg 后端，这样可以在普通 Python 脚本中弹出窗口显示图片
matplotlib.use('TkAgg')

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
nn.init.normal_(net[1].weight, mean=0, std=0.01)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()