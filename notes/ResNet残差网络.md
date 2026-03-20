#深度学习 #DeepLearning #ResNet
# 1. 残差块
**残差块（Residual Block）** 是深度残差网络（ResNet）的基础组件，它的提出主要是为了解决深度网络中的网络退化问题（即网络越深，`train_loss`反而越大的现象）.
![[Pasted image 20260320144812.png]]

> **残差块的核心思想**

给定输入$x$，一个神经网络层总是想要学习到理想的输出$H(x)$（即学习到真实解）.
- **传统网络**-试图让网络层直接逼近并学习这个复杂的函数$H(x)$.
- **残差网络**-让网络去学习输入和输出之间的差值$F(x)= H(x) -x$.最后再加上$x$就得到了预期的输出.

> **为什么这样做更好？**

1. 在极端情况下，如果当前层已经是最优的（已学习到真实解），再增加层数就不应该改变输出，所以后续的层都应当学到恒等映射.但是对于传统的非线性网络来说，要把权重训练成恒等映射是非常困难的.而对于残差网络来说，只需将残差$F(x)$的权重优化为0即可实现恒等映射，这一点更容易做到.
2. 将输入$x$直接跨过网络加到输出上，称为**捷径连接**或**跳跃连接**.这种设计在反向传播中缓解了梯度消失问题.因为在残差结构$H(x) = F(x) + x$中， 求导后总有一个常数`1`能够把梯度无损地传递给上一层.
3. 从数学计算的角度来看，只要输入$x$的维度`(in_channels,h,w)`与残差块的输出$F(x)$的维度相同，就可以直接相加.如果维度不同，只需要对$x$再加上一个$1\times 1$的卷积层，将指定维度与$F(x)$对齐即可.
4. 浅层的输入$x$保留了较多原始的图像细节和低级特征.将其跨过中间的非线性变换并与深层的高级语义特征$F(x)$相加，本质上是一种特征融合.

> 残差块的实现

**Input**
```python
import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

下面查看经过残差块后输入与输出的形状一致.
**Input**
```python
block = Residual_Block(3,3)
X = torch.rand(4, 3, 6, 6)
print(f'X的形状 = ', X.shape)
print(f'X直接经过残差块后的输出的形状 = ', block(X).shape)
```

**Output**
```python
X的形状 =  torch.Size([4, 3, 6, 6])
X直接经过残差块后的输出的形状 =  torch.Size([4, 3, 6, 6])
```

增加输出的通道数为3，并启用$1\times 1$卷积层，步幅为2，将宽和高减半.
**Input**
```python
block = Residual_Block(3,6, use_1x1conv=True, strides=2)
X = torch.rand(4, 3, 6, 6)
print(f'增加通道数，但减半宽和高 = ', block(X).shape)
```

**Output**
```python
增加通道数，但减半宽和高 =  torch.Size([4, 6, 3, 3])
```

# 2. ResNet模型 
![[Pasted image 20260320151600.png]]
ResNet的前两层与GoogLeNet一样，只不过在每一个卷积后都进行了批量规范化.
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```
而后接上`4`个**残差模块**（`ResNet_Block`），每一个残差模块中都有`2`个**残差块**(`Residual_Block`).
`ResNet_Block`通过控制`bool`类型变量`first_block`来决定是否改变`out_channels`.具体来说，第一个`ResNet_Block`的首个`Residual_Block`保持`in_channels = out_channels`，后面三个`ResNet_Block`的首个`Residual_Block`都会让`out_channels`变为原来的`2`倍.
```python
def ResNet_Block(in_channels, out_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual_Block(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual_Block(out_channels, out_channels))
    return block
    
b2 = nn.Sequential(*ResNet_Block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*ResNet_Block(64, 128, 2))
b4 = nn.Sequential(*ResNet_Block(128, 256, 2))
b5 = nn.Sequential(*ResNet_Block(256, 512, 2))
```

最后加入自适应平均汇聚层和全连接层输出.
```python
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

在训练之前，先观察ResNet的每一层形状是如何变化的.
**Input**
```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

**Output**
ResNet的特点是不断降低分辨率，提高通道数，最后用平均汇聚层聚集所有特征.
```python
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 128, 28, 28])
Sequential output shape:     torch.Size([1, 256, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape:      torch.Size([1, 512, 1, 1])
Flatten output shape:        torch.Size([1, 512])
Linear output shape:         torch.Size([1, 10])
```

> **在Fashion-MNIST数据集上进行训练.**

**Input**
```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
```

**Output**
```python
loss 0.014, train acc 0.996, test acc 0.878
2441.9 examples/sec on cuda:0
```
![[所有图片/深度学习图/现代卷积神经网络/Figure_5.png]]

> **对ResNet采取“批量规范化-激活-卷积”的架构会对性能有所改良，详见[[DenseNet稠密连接网络]].**