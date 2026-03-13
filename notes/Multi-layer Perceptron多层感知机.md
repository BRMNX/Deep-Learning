#深度学习 #DeepLearning #MLP #多层感知机
# 1. 隐藏层
```python
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
```
在线性回归中，我们都是直接使用全连接层来获取输出，也就是输出直接受到输入的影响。通过在中间引入**隐藏层**，每一层的结果继续输入给上面的层，这样的架构称为**多层感知机（Multilayer Perceptron）**.
![[Pasted image 20260309221207.png]]
很明显，只用线性模型来进行运算具有很大的局限性，比如当体温高于37°C或低于37°C时都应该产生异常，但线性模型做不到这一点.引入隐藏层不光是为了让模型更复杂，也是为了引入非线性性.
假定输入$X \in \mathbb{R}^{n\times d}$表示一个包含$n$个样本的小批量，每个样本有$d$个输入特征（维度）.输出$O \in \mathbb{R}^{n\times q}$表示$q$个类别.利用下面的公式来计算$$\mathbf{H = XW^{(1)}+b^{(1)}},$$$$\mathbf{O = HW^{(2)}+b^{(2)}}.$$其中$W^{(1)}\in \mathbb{R}^{d\times h},H\in\mathbb{R}^{n\times h},W^{(2)}\in\mathbb{R}^{h\times q}$.但是这样的运算是愚蠢的，因为两个线性函数的符合仍然是线性函数.为了发挥多层架构的潜力，还需要一个关键要素：非线性激活函数$\sigma$，使得算法变换为$$\mathbf{H = \sigma(XW^{(1)}+b^{(1)})},$$$$\mathbf{O = HW^{(2)}+b^{(2)}}.$$一般来说，有了激活函数，MLP就不太可能再坍缩成一个线性模型了.
# 2. 激活函数
```python
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
```
## 2.1 ReLU函数
最受欢迎的激活函数是**修正线性单元ReLU**，它不仅实现简单，而且预测性能良好.$$\mathrm{ReLU}(x) = \max\{0,x\}.$$
**Input**
```python
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'ReLU(x)', figsize=(5,2.5))
    plt.show()
    
```
**Output（ReLU(x)的图象）**
![[Pasted image 20260309222259.png]]
规定在$x=0$处ReLU函数的导数值为0.
**Input**
```python
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of ReLU(x)', figsize=(5, 2.5))
    plt.show()
```
**Output（ReLU(x)的梯度）**
- ReLU函数的梯度**始终存在**，因此**不会发生梯度消失**！

![[Figure_14.png]]
ReLU函数还有许多变体，比如**参数化ReLU（pReLU）函数**，它为ReLU函数增加一个线性项，使得参数为负时也可以通过$$\mathrm{pReLU}(x) = \max\{0,x\}+\alpha\min\{0,x\}.$$
## 2.2 Sigmoid函数
Sigmoid是从$\mathbb{R}$到$(0,1)$的函数，通常被称为挤压函数.$$\mathrm{sigmoid(x)} = \frac{1}{1+\exp(-x)}.$$
用上面同样的代码（除了改变`y = torch.sigmoid(x)`以外）可以绘制出它的图象：
![[Pasted image 20260309223619.png]]
以及它的梯度$$\mathrm{sigmoid'(x)} = \mathrm{sigmoid(x)}(1-\mathrm{sigmoid}(x)),$$
![[Pasted image 20260309223636.png]]
## 2.3 Tanh函数
这是数学上常见的**双曲正切函数tanh**，它将$\mathbb{R}$映射到$(-1,1)$，与sigmoid类似但略有不同.$$\tanh(x) = \frac{1-\exp(-2x)}{1+\exp(-2x)}.$$
它关于原点对称：
![[Pasted image 20260309223832.png]]
梯度：$$\tanh'(x) = 1-\tanh^{2}(x),$$![[Pasted image 20260309224027.png]]
# 3. 多层感知机实现
## 3.1 手动实现
实现有一个隐藏层，256个隐藏单元的MLP.
**Input**
```python
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
d2l.predict_ch3(net, test_iter) # 输出一些预测
plt.show()
```
在创建`W1,b1,W2,b2`时，使用`nn.Parameter()`会自动将`requires_grad`设为`True`，使优化器（Optimizer）跟踪这些变量从而计算梯度.
**Output**
![[所有图片/深度学习图/多层感知机/Figure_1.png]]
![[所有图片/深度学习图/多层感知机/Figure_2.png]]
## 3.2 简洁实现
实现两个全连接层，其中第一层的输出经过ReLU之后传给下一层.与手动实现相比，不需要自己定义参数`W1,b1,W2,b2`，不需要写`net`和`relu`方法，其他类似.
**Input**
```python
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
```
`init_weights()`方法便于初始化每一层的权重（利用`net.apply(init_weights)`），而不需要手动`nn.init.normal_(net[1].weight)`和`nn.init.normal_(net[3].weight)`.
==**源码解析**==
`nn.Sequential`类是一个**顺序容器（sequential container）**，当它的构造函数被调用时，模型会被按序加入其中.`nn.Sequential` 允许你将多个层（如展平、线性变换、激活函数）按顺序排列.它会自动将**前一个层的输出作为下一个层的输入**.
- **存储结构**：`Sequential`继承自`Module`，内部使用一个有序字典`OrderedDict`来存储传入的模块.如果直接传入参数（也就是每一层，比如这里的`nn.Sequential(nn.Flatten(),nn.Linear(),nn.ReLU(),nn.Linear())`），它会按顺序给它们编号（0,1,2,...）并注册为子模块.
- **前向传播逻辑**（`forward()`）：其核心逻辑非常简单，源码如下
```python
def forward(self, input): 
	for module in self: 
		input = module(input) 
	return input
```
它遍历容器中的每一个子模块`module`，将输入数据`input`依次传递给它们，并将当前模块的输出作为下一个模块的输入，最终返回最后一个模块的输出.这里的`input`通常是**一个批次的样本数据`X`**，具体表现为一个**Tensor**，大小为`(batch_size, num_inputs)`.
**Output**
![[所有图片/深度学习图/多层感知机/Figure_3.png]]
## 3.3 问题（待修改）
> [!question] 
> 改变隐藏单元数`num_hiddens`,并绘制其数量与准确率的图象.这个超参数的最佳值是多少？（注意：`num_hiddens`通常取2的较大幂次）

**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),nn.LazyLinear(num_hiddens),nn.ReLU(),nn.LazyLinear(num_outputs))

def get_accuracy(data_loader, model):
    total_num = 0
    total_acc = 0
    for X,y in data_loader:
        y_hat = model(X)
        batch_size = X.shape[0]
        batch_acc = model.accuracy(y_hat, y, averaged=True) * batch_size
        total_num += batch_size
        total_acc += batch_acc
    return total_acc / total_num

def evaluate_acc_with_different_hiddens(num_hiddens):
    val_acc = []
    data = d2l.FashionMNIST(batch_size=256)
    for num_h in num_hiddens:
        print(f'h = ',num_h)
        model = MLP(num_outputs=10,num_hiddens=num_h,lr=0.05)
        trainer = d2l.Trainer(max_epochs=10)
        trainer.fit(model,data)
        val_loader = data.get_dataloader(train=False)
        # 计算此轮次的准确率
        epoch_acc = get_accuracy(val_loader, model) 
        val_acc.append(epoch_acc.detach().numpy()
    return val_acc

if __name__ == '__main__':
    num_hiddens = [64,128,256,512,1024,2048]
    val_acc = evaluate_acc_with_different_hiddens(num_hiddens)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(num_hiddens)), val_acc, tick_label=num_hiddens, color='skyblue', edgecolor='black')
    # 添加标题和标签
    plt.title('Validation Accuracy vs. Number of Hidden Units', fontsize=14)
    plt.xlabel('Number of Hidden Units (num_hiddens)', fontsize=12)
    plt.ylabel('Validation Accuracy (val_acc)', fontsize=12)
    # 添加网格线 (仅Y轴，方便读数)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 自动调整布局以防标签被截断
    plt.tight_layout()
    # 显示图表
    plt.show()
```
这里`num_hiddens = [64,128,256,512,1024,2048]`，整个训练过程耗时10~20分钟.
**Output**
![[Figure_17.png]]
没有看出哪个值有明显的优势（可能是隐藏单元数太多，或是纵坐标尺度没有调整以至于差别不明显）.
> [!question] 
> 再添加一个隐藏层，查看结果.

在`MLP`类中加入一个隐藏层，仍有256个隐藏单元.
**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),nn.LazyLinear(num_hiddens),nn.ReLU(),nn.LazyLinear(num_hiddens),nn.ReLU(),nn.LazyLinear(num_outputs))

if __name__ == '__main__':
    data = d2l.FashionMNIST(batch_size=256)
    model = MLP(num_outputs=10,num_hiddens=256,lr=0.05)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model,data)
    plt.show()
```
**Output**
![[Figure_18.png]]
最终的准确率在$83\%$左右.这与单隐藏层的结果差别不大，因为`FashionMNIST`只是一个小数据集，单层256单元已经足够捕捉它的特征.
> [!question]
> 为什么只插入一个神经元的隐藏层是不好的？会有什么问题？

代码运行前的猜测：单个神经元可能会过度融合样本信息，导致特征信息丢失.结果是训练误差和验证误差都很大，模型完全无法有效拟合.
**Input**
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),nn.LazyLinear(num_hiddens),nn.ReLU(),nn.LazyLinear(num_outputs))

if __name__ == '__main__':
    data = d2l.FashionMNIST(batch_size=256)
    model = MLP(num_outputs=10,num_hiddens=1,lr=0.05)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model,data)
    plt.show()
```
**Output**
![[Figure_19.png]]
事实证明，**模型完全无法工作**.这是因为输出层实际上是对ReLU函数的结果（单个神经元的结果）再进行一次线性变换，但ReLU本身是分段线性，再经过线性变换仍然是分段线性的.这实际上等价于一个单层线性分类器，对于`FashionMNIST`这种多分类任务，一个神经元根本无法提取有效特征.对于10个类别的分布，随机猜测的交叉熵损失$$-\log\left(\frac{1}{10}\right) = 2.3026,$$恰好是模型的损失.也就是说，模型输出几乎是**均匀分布**（每个类别的概率都为0.1），完全没有学习能力.