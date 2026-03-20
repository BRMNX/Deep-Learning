import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.BatchNorm2d(in_channels),nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                out_channels * i + in_channels, out_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

# block = DenseBlock(num_convs=2, in_channels=3, out_channels=10)
# X = torch.randn(4, 3, 8, 8)
# print(f'经过DenseBlock后输出的形状 = ', block(X).shape)

def TransitionBlock(in_channels, out_channels): # 我知道这样的函数命名不规范
    return nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2))
# block1 = DenseBlock(num_convs=2, in_channels=3, out_channels=10)
# block2 = TransitionBlock(23, 10)
# print(f'先经过稠密块，再过过渡层TransitionBlock后输出的形状 = ', block2(block1(X)).shape)

# DenseNet实现
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
out_channels = 64
growth_rate = 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blocks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blocks.append(DenseBlock(num_convs = num_convs, in_channels=out_channels, out_channels=growth_rate))
    out_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blocks.append(TransitionBlock(in_channels=out_channels, out_channels=out_channels // 2))
        out_channels = out_channels // 2
net = nn.Sequential(
    b1, *blocks,
    nn.BatchNorm2d(out_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(out_channels, 10))
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256, resize=96)
lr, num_epochs, device = 0.1, 10, d2l.try_gpu()
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
plt.show()