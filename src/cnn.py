import torch
from torch import nn
from d2l import torch as d2l

import numpy as np
import matplotlib.pyplot as plt

def corr2d(X:torch.Tensor, K:torch.Tensor):
    Y = torch.zeros((X.shape[0]-K.shape[0]+1, X.shape[1]-K.shape[1]+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+K.shape[0],j:j+K.shape[1]]*K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias 
    
def section1():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = corr2d(X, K)
    print(Y)

def section2():
    # 边缘检测
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    # 卷积层
    X = X.reshape((1, 1, 6, 8))
    K = K.reshape((1, 1, 1, 2))
    lr = 3e-2
    # in_channels是输入图像的通道数
    # out_channels是经过卷积后预期的输出图像通道数
    net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (1, 2), bias = False)
    for i in range(10):
        Y_hat = net(X)
        l = (Y_hat - Y) ** 2
        net.zero_grad()
        l.sum().backward()
        net.weight.data -= lr * net.weight.grad
        if (i+1)%2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')
    print(f'学习后的卷积核K = ', net.weight.data.reshape((1, 2)))

def question1():
    X = torch.eye(8)
    K = torch.tensor([[1.0,-1.0]])
    print(f'Diagonal matrix X = \n{X}')
    print(f'Apply corr2d(X, K) = \n{corr2d(X, K)}')
    print(f'Apply corr2d(X.t(), K) = \n{corr2d(X.t(), K)}')
    print(f'Apply corr2d(X, K.t()) = \n{corr2d(X, K.t())}')

def comp_conv2d(net, X):
    X = X.reshape((1,1) + X.shape)
    # Y形如(1, 1, h, w)，其中h和w分别是输出的高和宽
    # Y.shape[2:]只保留h和w，舍去Y的batch_size和out_channels维度
    # 但是只有在总元素个数相同时(也就是被舍去的轴维度都是1)才能这样reshape
    Y = net(X)
    return Y.reshape(Y.shape[2:])

def section4():
    # 这里padding=1让每边都填充了了一行或一列，一共填充2行2列
    net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding=1)
    X = torch.rand(size=(8,8))
    # 查询输出的大小
    print(comp_conv2d(net, X).shape)
    # 步幅为2
    net = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding=1, stride=2)
    print(comp_conv2d(net, X).shape)

def corr2d_multi_in(X, K):
    # zip()函数把X的每个通道和K的对应通道一一配对，然后for循环遍历每个通道
    # 每次返回的x都是该通道上的二维张量，k是该通道上的二维卷积核
    # 默认情况下，zip()函数以第0个维度的长度为准进行配对，也就是通道数
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
def corr2d_multi_in_out_1x1(X,K):
    c_i, h, w = X.shape
    c_0 = K.shape[0]
    K = K.reshape(c_0, c_i)
    X = X.reshape(c_i, h*w)
    Y = torch.matmul(K, X)
    return Y.reshape(c_0, h, w)
def section5():
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(f'2通道输入的corr2d(X,K) = \n{corr2d_multi_in(X,K)}')
    K = torch.stack((K,K+1,K+2),0)
    print(f'2通道输入3通道输出的corr2d(X,K) = \n{corr2d_multi_in_out(X,K)}')
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    print(f'1x1卷积核的输出 = \n{Y1}')
    print(f'一般卷积核的输出 = \n{Y2}')
    print(f'是否认为它们相等:{abs((Y1- Y2).sum()) < 1e-6}')
def pool2d(X,pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i:i+p_h,j:j+p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i+p_h,j:j+p_w].mean()
    return Y
def section6():
    X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
    print(f'2x2最大汇聚层的输出 = \n{pool2d(X,(2,2))}')
    X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))
    print(f'X = \n{X}')
    pool = nn.MaxPool2d(3)
    print(f'3x3最大汇聚层的输出 = \n{pool(X)}')
    pool = nn.MaxPool2d(3, padding=1, stride=2)
    print(f'3x3最大汇聚层的输出(步幅2，填充1) = \n{pool(X)}')
    # 在输入通道上添加X+1，X+2，形成3通道输入
    X = torch.cat((X, X+1, X+2),1)
    pool = nn.MaxPool2d(3, padding=1, stride=2)
    print(f'3x3最大汇聚层3通道输入的输出(步幅2，填充1) = \n{pool(X)}')
def edge_detection():
    img_path = "xxx"
    img = plt.imread(img_path)
    img_tensor = torch.tensor(img,dtype=torch.float32)
    h, w = img_tensor.shape
    kernel = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]])
    Y1 = corr2d(img_tensor, kernel)
    Y2 = corr2d(img_tensor, kernel.t())
    Y3 = 0.5 * Y1 + 0.5 * Y2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
    axes[0].imshow(Y1, cmap='gray')
    axes[0].set_title('Vertical Edge Detection')
    axes[1].imshow(Y2, cmap='gray')
    axes[1].set_title('Horizontal Edge Detection')
    axes[2].imshow(Y3, cmap='gray')
    axes[2].set_title('Mixed Edge Detection')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # section1()
    # section2()
    # question1()
    # section4()
    # section5()
    # section6()
    # edge_detection()
    print('Done.')