#深度学习 #DeepLearning #规范化
# 1. 批量规范化
从形式上说，批量规范化类似于随机变量的标准正态化，对于一个批次$\mathcal B$的样本$\mathbf{x}$，减去批次的均值并除以标准差，然后再用一个线性变换，就得到批量规范化BN：$$\mathrm{BN}(\mathbf{x}) = \gamma \circ \frac{\mathbf{x-\mu_{\mathcal B}}}{\sigma_{\mathcal B}} + \beta.$$这里的圆圈符号是按元素运算，其中$$\mu_{\mathcal B} = \frac{1}{|\mathcal B|}\sum_{\mathbf{x}\in \mathcal B} \mathbf{x},\sigma_{\mathcal B}^{2} = \sum_{\mathbf{x}\in \mathcal B}(\mathbf{x - \mu_{\mathcal B}})^{2}+\varepsilon.$$拉伸参数$\gamma$和偏移参数$\beta$是学习的参数，形状与$\mathbf{x}$相同，通常初始化$\gamma = 1,\beta = 0$.$\varepsilon$是给定的噪声.在方差中添加噪声是为了防止分母为0.
# 2. 批量规范化层
## 2.1 全连接层
设全连接层的输入为$\mathbf{x}$，权重和偏置为$\mathbf{W}$和$\mathbf{b}$，激活函数为$\sigma$，那么使用批量规范化的全连接层输出为$$\mathbf{h} = \sigma(\mathrm{BN}(\mathbf{Wx+b})).$$
## 2.2 卷积层
当卷积层有多个输出通道时，需要对每个通道执行批量规范化，每个通道都有自己的拉伸参数$\gamma$和偏移参数$\beta$.假定批量中有$m$个样本，并且对于每个通道，卷积的输出是`(p,q)`，那么每个输出通道的批量规范化都需要考虑$m\times p \times q$个元素的均值和方差.
## 2.3 训练模式与验证模式
实际上批量规范化会维护两个变量：全局移动均值$\mu_{\mathrm{global}}$和全局移动方差$\sigma^{2}_{\mathrm{global}}$，它们的初值为`0`和`1`.

当模型处于`train`模式时，模型按上面的算法计算每个批次的$\mu_{\mathcal B}$和$\sigma^{2}_{\mathcal B}$，并通过一个动量参数（momentum）$\alpha$更新全局平均和方差：$$\mu_{\mathrm{global}}\leftarrow \alpha\mu_{\mathrm{global}} + (1-\alpha)\mu_{\mathcal B},$$$$\sigma^{2}_{\mathrm{global}} \leftarrow \alpha\sigma^{2}_{\mathrm{global}} + (1-\alpha)\sigma^{2}_{\mathcal B}$$
当模型处于`eval`模式时，对每个批量直接应用$\mu_{\mathrm{global}}$和$\sigma^{2}_{\mathrm{global}}$来进行批量规范化，并且不再更新它们（因为验证模式下，输入相同的样本应该得到相同的输出，参数在此情况下不能再更新）.

> **补充说明：** 全局均值和方差在深度学习框架（如 PyTorch）中并不被视为“模型参数”（Parameters），而是被称为**缓冲区状态（Buffers）**.它们不参与反向传播的梯度更新，只在训练的前向传播过程中按照给定的动量（Momentum）默默进行移动平均的累加.

# 3. 带有批量规范化层的LeNet实现
注意仅规范化是不包含激活函数的.
**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
    # 只有当X是全连接层或卷积层的输入才执行
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2: # 全连接层,X.shape = (batch_size, num_features)
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim = 0)
        else: # 卷积层,X.shape = (batch_size, num_channels, height, width)
            mean = X.mean(dim=(0,2,3), keepdim=True) # (1, num_channels, 1, 1)
            var = ((X - mean)**2).mean(dim=(0,2,3), keepdim=True) # (1, num_channels, 1, 1)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1) 
        # gamma与beta的值仅在各通道上不同，计算时调用广播机制
        self.gamma = nn.Parameter(torch.ones(shape)) 
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

# 用于LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
print(f'第一个规范化层学习到的参数：')
print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
print(f'第二个规范化层学习到的参数：')
print(net[5].gamma.reshape((-1,)), net[5].beta.reshape((-1,)))
```

**Output**
![[所有图片/深度学习图/现代卷积神经网络/Figure_4.png]]
```python
loss 0.263, train acc 0.902, test acc 0.832
36547.8 examples/sec on cuda:0
第一个规范化层学习到的参数：
# gamma
tensor([2.5840, 3.7791, 2.8622, 3.3028, 2.2195, 2.4231], device='cuda:0',
       grad_fn=<ReshapeAliasBackward0>) 
# beta
tensor([ 2.3216, -2.2555, -1.2703,  3.3450,  1.8077,  1.8018], device='cuda:0',
       grad_fn=<ReshapeAliasBackward0>)
第二个规范化层学习到的参数：
# gamma
tensor([2.0597, 0.9051, 2.0675, 1.8783, 1.3954, 1.9372, 2.2877, 0.4641, 2.4002,
        1.3843, 1.8694, 2.0692, 2.2286, 1.3702, 1.8499, 1.7879],
       device='cuda:0', grad_fn=<ReshapeAliasBackward0>) 
#beta       
tensor([-0.1714, -0.5306, -0.1101,  0.6538,  0.3872, -0.3188,  0.1261, -0.0634,
         0.9395,  0.5247,  0.7768,  0.1204,  0.8133,  0.2889, -0.0666,  0.2309],
       device='cuda:0', grad_fn=<ReshapeAliasBackward0>)
```

**调用高级API**
也可以在`nn.Sequential`中直接使用内置的`nn.BatchNorm2d(num_features)`，其唯一的参数是输出通道数`num_features`，它会自适应上一层是全连接/卷积，不需要指定.