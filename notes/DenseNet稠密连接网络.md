#深度学习 #DeepLearning #DenseNet

> **从ResNet到DenseNet**

ResNet将函数展开为两部分$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}),$$一个简单线性项和一个非线性项.如果想将$f(\mathbf{x})$展开成更多部分，就产生了一种方案DenseNet.
![[Pasted image 20260320163639.png]]
上图的左边是ResNet，右边是DenseNet.它们的显著区别是，DenseNet不再将输入与输出直接相加，而是将它们拼接起来，形成**稠密连接**.$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$
![[Pasted image 20260320163902.png]]

> **DenseNet由稠密块（DenseBlock）与过渡层（TransitionBlock）组成.** 下面将依次介绍.
# 1. DenseBlock稠密块
一个稠密块由多个卷积块组成，每个卷积块都使用了ResNet改良版的“批量规范化-激活-卷积”架构.**该卷积块保持输入与输出的大小一致，但可能会改变通道数**.
```python
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
```

然后，我们实现`DenseBlock`.

```python
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
```
>==**源码解析**==
>在`DenseBlock`的构造函数中，我们向`layer`中添加卷积块，每一个卷积块的输入维度是`out_channels * i + in_channels`，输出维度固定为`out_channels`，它们的前后维度是不匹配的.但是`DenseBlock`的核心是连结，而不是按顺序计算.在下面的`forward`方法中，会遍历`net`中的每一个卷积块`block`.
>
>假设`X`的输入通道是`in_channels`，`DenseNet`预期的输出通道数是`out_channels`.
>
>对于第一个卷积块（`i=0`），输入通道需求为`out_channels * 0 + in_channels = in_channels`，恰好为输入`X`的通道数.输出通道为`out_channels`，因此`Y`的形状为`out_channels`.
>
>**关键操作**：`X = torch.cat((X,Y), dim=1)`，它将`X`和`Y`在通道维度上连结在一起，拼接后，输入通道数就变成了`out_channels * 1 + in_channels`，恰好符合下一个卷积块的输入通道数.依此类推，就可以理解为什么要用`out_channels * i + in_channels`了.

通过`DenseBlock`后，容易算出最终所得输出的通道数是`in_channels + num_convs * out_channels`.
**Input**
```python
block = DenseBlock(num_convs=2, in_channels=3, out_channels=10)
X = torch.randn(4, 3, 8, 8)
print(f'经过DenseBlock后输出的形状 = ', block(X).shape)
```
**Output**
```python
经过DenseBlock后输出的形状 =  torch.Size([4, 23, 8, 8])
```

# 2. TransitionBlock过渡层
由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型. 而过渡层可以用来控制模型复杂度. 它通过卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度.
```python
def TransitionBlock(in_channels, out_channels): # 我知道这样的函数命名不规范
    return nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=1),
                         nn.AvgPool2d(kernel_size=2, stride=2))
```

**Input**
```python
X = torch.randn(4, 3, 8, 8)
block1 = DenseBlock(num_convs=2, in_channels=3, out_channels=10)
block2 = TransitionBlock(23, 10)
print(f'先经过稠密块，再过过渡层TransitionBlock后输出的形状 = ', block2(block1(X)).shape)
```

**Output**
```python
先经过稠密块，再过过渡层TransitionBlock后输出的形状 =  torch.Size([4, 10, 4, 4])
```

# 3. DenseNet模型
DenseNet的第一部分与ResNet一致.
```python
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

接下来使用4个稠密块，每个稠密块中有4个卷积层（`num_convs = 4`），每个卷积层会使通道数增加32（增长率`growth_rate`）.在每个稠密块之间，用过渡层来减半宽、高和通道数.
```python
out_channels = 64 # 注意辨别out_channels在不同方法中的不同含义
growth_rate = 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blocks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blocks.append(DenseBlock(num_convs = num_convs, in_channels=out_channels, out_channels=growth_rate))
    out_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blocks.append(TransitionBlock(in_channels=out_channels, out_channels=out_channels // 2))
        out_channels = out_channels // 2
```

与ResNet类似，最后接上全局汇聚层和全连接层来输出结果.
```python
net = nn.Sequential(
    b1, *blocks,
    nn.BatchNorm2d(out_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(out_channels, 10))
```

> **训练DenseNet模型.**

**Input**
```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256, resize=96)
lr, num_epochs, device = 0.1, 10, d2l.try_gpu()
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
plt.show()
```

**Output**
```python
loss 0.139, train acc 0.949, test acc 0.863
2464.3 examples/sec on cuda:0
```
![[所有图片/深度学习图/现代卷积神经网络/Figure_6.png]]